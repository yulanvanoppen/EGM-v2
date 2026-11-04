classdef MAGI < handle                                                  % Manifold-constrained Gaussian process inference
%MAGI wraps the MAGI method of Yang, Wong and Kou (2021), PNAS 118(15),
%e2020397118, as implemented by the authors at github.com/wongswk/magi.
%
%Unlike the other estimators in this package, MAGI is NOT reimplemented
%here. The sampler (Matern GP covariances, hyperparameter optimization and
%the Hamiltonian Monte Carlo scheme) lives in the compiled C++ library
%libcmagi, reached through the solveMagi MEX file; this class only adapts
%the EGM system and data to the MagiSolver interface and maps the posterior
%back onto the usual EGM output fields. See dependencies() below for what
%has to be on the MATLAB path.
%
%The adapter is exact rather than numerical. MAGI asks for three functions,
%    fOde(theta, x, t)        n-by-d
%    fOdeDx(theta, x, t)      n-by-d-by-d, entry (:,j,i) = df_i/dx_j
%    fOdeDtheta(theta, x, t)  n-by-p-by-d, entry (:,q,i) = df_i/dtheta_q
%and System already provides all three: fOde is rhs, fOdeDx is df, and
%because the EGM systems are linear in the parameters, f = g(x)*beta + h(x)
%gives fOdeDtheta = g(x) analytically. Only the stacked K blocks that
%System returns have to be rearranged into MAGI's 3-D layout.
%
%MAGI infers trajectories for every component, including completely
%unobserved ones, which enter as all-NaN columns of the data matrix. This
%is a different treatment of hidden states from GP: they are sampled under
%the GP prior and the manifold constraint rather than profiled out.
%
%Two mismatches with the rest of the package are worth knowing about:
%  - MAGI assumes additive Gaussian measurement error with one standard
%    deviation per component, whereas EGM uses sigma2 + tau2*x^2. With
%    strongly multiplicative noise the MAGI noise model is misspecified.
%  - MAGI infers a single trajectory, so for N > 1 cells only the first is
%    used, and a warning is issued.
%
%See also EGM, GP, TM, MAGISOLVER.

    properties (SetAccess = private)
        data                                                                % data aggregate struct
        system                                                              % nested object controlling the ODE system
        settings                                                            % hyperparameters and user input

        T                                                                   % number of data time points
        L                                                                   % number of observed states
        N                                                                   % population size
        K                                                                   % system dimension
        P                                                                   % number of model parameters

        observed                                                            % observed/hidden state indices
        hidden

        y                                                                   % MAGI data matrix, NaN where unobserved
        tvec                                                                % discretization set I
        model                                                               % odeModel struct passed to MagiSolver
        control                                                             % control struct passed to MagiSolver
        raw                                                                 % unmodified MagiSolver output

        beta_magi                                                           % model parameter estimate
        varbeta                                                             % posterior covariance of beta
        beta_interval                                                       % credible interval for beta
        sigma_magi                                                          % measurement error SD estimate
        phi                                                                 % estimated GP hyperparameters

        smoothed_magi                                                       % posterior mean trajectory on tvec
        dsmoothed_magi
        trajectory_interval                                                 % pointwise credible band
    end


    methods
        function obj = MAGI(data, system, settings, beta_init, x_init) % Constructor
            obj.data = data;                                                % experimental data
            obj.system = system;                                            % ODE system

            defaults = struct('level', 1, 'niter', 20000, 'nsteps', 200, ...% MagiSolver defaults, mirrored here so that a
                              'burnin', .5, 'band', 20, 'stepsize', [], ... % settings struct assembled by hand still works
                              'kernel', 'generalMatern', 'sigma', [], ...
                              'fixed_sigma', false, 'verbose', false, ...
                              'summary', 'mean', 'init_mode', 'magi', ...
                              'positive', true, 'lb', [], 'ub', []);
            fields = fieldnames(defaults);
            for f = 1:numel(fields)
                if ~isfield(settings, fields{f})
                    settings.(fields{f}) = defaults.(fields{f});
                end
            end
            obj.settings = settings;                                        % hyperparameters

            [obj.T, obj.L, obj.N] = size(data.traces);
            obj.K = system.K;
            obj.P = system.P;

            obj.observed = reshape(data.observed, 1, []);
            obj.hidden = find(~ismember(1:obj.K, obj.observed));

            obj.beta_magi = reshape(beta_init, 1, []);
            obj.build_data(x_init);
            obj.build_model();
            obj.build_control(beta_init, x_init);
        end


        function output = optimize(obj)                                 % Main routine: hand over to MagiSolver
            MAGI.dependencies();                                            % fail early and legibly

            obj.raw = MagiSolver(obj.y, obj.model, obj.tvec, obj.control);

            obj.save_estimates();
            output = obj.data;                                              % return data appended with MAGI results
        end


        function build_data(obj, x_init)                                % Data matrix over the discretization set I
            if obj.N > 1
                warning('EGM:MAGI:replicates', ...                          % MAGI infers one trajectory, not a population
                        'MAGI uses a single trajectory; cell 1 of %d is used.', obj.N)
            end

            observations = NaN(obj.T, obj.K);                               % unobserved components are all-NaN columns,
            observations(:, obj.observed) = obj.data.traces(:, :, 1);       % which MAGI infers under the GP prior

            filled = MAGI.discretize([reshape(obj.data.t, [], 1) observations], obj.settings.level);
            obj.tvec = reshape(filled(:, 1), 1, []);
            obj.y = filled(:, 2:end);

            obj.data.tvec_magi = obj.tvec;
            if nargin > 1 && ~isempty(x_init), obj.data.init_magi = reshape(x_init, 1, []); end
        end


        function build_model(obj)                                       % Adapt System to the MagiSolver interface
            sys = obj.system;
            dimension = obj.K;
            parameters = obj.P;

            obj.model.fOde = @(theta, x, t) sys.rhs(x, reshape(t, 1, []), reshape(theta, 1, []));

                                                                            % System stacks K blocks of n rows; MAGI wants
                                                                            % n-by-d-by-d with (:,j,i) = df_i/dx_j, so
                                                                            % reshape the stack and swap the last two axes
            obj.model.fOdeDx = @(theta, x, t) permute(reshape( ...
                sys.df(x, reshape(t, 1, []), reshape(theta, 1, [])), ...
                size(x, 1), dimension, dimension), [1 3 2]);

                                                                            % f = g(x)*beta + h(x) is linear in beta, so
                                                                            % df_i/dtheta_q is exactly g, rearranged the
                                                                            % same way into n-by-p-by-d
            obj.model.fOdeDtheta = @(theta, x, t) permute(reshape( ...
                sys.g(x, reshape(t, 1, [])), ...
                size(x, 1), dimension, parameters), [1 3 2]);

            lb = obj.settings.lb;
            ub = obj.settings.ub;
            if isempty(lb), lb = zeros(1, obj.P); end
            if isempty(ub), ub = Inf(1, obj.P); end
            obj.model.thetaLowerBound = reshape(lb, 1, []);
            obj.model.thetaUpperBound = reshape(ub, 1, []);
        end


        function build_control(obj, beta_init, x_init)                  % MagiSolver control struct
            obj.control = struct();
            obj.control.niterHmc = obj.settings.niter;
            obj.control.nstepsHmc = obj.settings.nsteps;
            obj.control.burninRatio = obj.settings.burnin;
            obj.control.bandSize = obj.settings.band;
            obj.control.kerneltype = char(obj.settings.kernel);
            obj.control.positiveSystem = logical(obj.settings.positive);
            obj.control.verbose = logical(obj.settings.verbose);
            obj.control.theta = reshape(beta_init, 1, []);

            if ~isempty(obj.settings.stepsize), obj.control.stepSizeFactor = obj.settings.stepsize; end
            if ~isempty(obj.settings.sigma)
                obj.control.sigma = reshape(obj.settings.sigma, 1, []);
                obj.control.useFixedSigma = logical(obj.settings.fixed_sigma);
            end

            if strcmpi(obj.settings.init_mode, 'integrate') && ~isempty(x_init)
                try                                                         % optional warm start from a single ODE solve,
                    int_data = obj.data;                                    % otherwise MAGI's own routine is used
                    int_data.init = reshape(x_init, 1, []);
                    obj.control.xInit = obj.system.integrate(reshape(beta_init, 1, []), int_data, obj.tvec);
                catch ME
                    disp(ME)
                end
            end
        end


        function save_estimates(obj)                                    % Summarize the posterior
            theta = obj.raw.theta;                                          % samples-by-P
            sampled = obj.raw.xsampled;                                     % samples-by-nI-by-K

            if strcmpi(obj.settings.summary, 'median')
                obj.beta_magi = median(theta, 1);
                trajectory = reshape(median(sampled, 1), size(sampled, 2), size(sampled, 3));
            else
                obj.beta_magi = mean(theta, 1);
                trajectory = reshape(mean(sampled, 1), size(sampled, 2), size(sampled, 3));
            end

            obj.varbeta = cov(theta);                                       % genuine posterior uncertainty, unlike the
            obj.beta_interval = quantile(theta, [.025 .975], 1);            % asymptotic approximations used elsewhere
            obj.sigma_magi = mean(obj.raw.sigma, 1);
            obj.phi = obj.raw.phi;

            obj.smoothed_magi = trajectory;
            obj.dsmoothed_magi = obj.system.rhs(trajectory, obj.tvec, obj.beta_magi);
            obj.trajectory_interval = quantile(sampled, [.025 .975], 1);

            init = trajectory(1, :);
            if obj.settings.positive, init = max(init, 1e-12); end

            obj.data.init = init;                                           % store results
            obj.data.beta_magi = obj.beta_magi;
            obj.data.varbeta = obj.varbeta;
            obj.data.beta_interval = obj.beta_interval;
            obj.data.sigma_magi = obj.sigma_magi;
            obj.data.phi_magi = obj.phi;
            obj.data.smoothed_magi = obj.smoothed_magi;
            obj.data.dsmoothed_magi = obj.dsmoothed_magi;
            obj.data.trajectory_interval = obj.trajectory_interval;
            obj.data.theta_samples = theta;
            obj.data.lp_magi = obj.raw.lp;
        end
    end


    methods (Static)
        function filled = discretize(matrix, level)                     % Insert discretization points (cf. setDiscretization)
            if level <= 0                                                   % 2^level - 1 points between each pair of rows
                filled = matrix;
                return
            end
            matrix = sortrows(matrix);
            inserted = NaN(size(matrix, 1) - 1, size(matrix, 2));
            inserted(:, 1) = (matrix(2:end, 1) + matrix(1:end-1, 1)) / 2;
            filled = MAGI.discretize(sortrows([matrix; inserted]), level - 1);
        end


        function dependencies()                                         % Check that the compiled MAGI library is reachable
            missing = {};
            if isempty(which('MagiSolver')), missing{end+1} = 'MagiSolver.m'; end
            if isempty(which('solveMagi')), missing{end+1} = 'solveMagi MEX file'; end
            if ~isempty(missing)
                error('EGM:MAGI:dependencies', ...
                      ['MAGI requires the compiled library from github.com/wongswk/magi. Missing: %s.\n' ...
                       'Add the MATLABmagi folder to the MATLAB path. On Windows the prebuilt binaries in\n' ...
                       'MATLABmagi/windows (solveMagi.mexw64, libcmagi.dll, libgcc_s_seh_64-1.dll,\n' ...
                       'libgfortran_64-3.dll) must sit in a folder on the path, together with libblas.dll\n' ...
                       'and liblapack.dll obtained separately from the LAPACK for Windows distribution.'], ...
                      strjoin(missing, ', '))
            end
        end
    end
end
