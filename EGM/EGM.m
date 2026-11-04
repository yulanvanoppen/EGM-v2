classdef EGM < handle
    
    properties (SetAccess = private)
        data                                                                % measurements struct
        system                                                              % ODE system object
        
        autoknots                                                           % logical for placement heuristic
        knots                                                               % B-spline knots for each state
        interactive                                                         % logical for interactive smoothing app use
        
        lb                                                                  % parameter space lower bounds
        ub                                                                  % parameter space upper bounds
        positive                                                            % logical for forced state positivity
        t_gm                                                                % first stage optimization grid (GMGTS)
        
        niterSM                                                             % #iterations for smoothing
        tolSM                                                               % convergence tolerance for smoothing
        niterGM                                                             % #iterations for the first stage
        tolGM                                                               % convergence tolerance for the first stage

        M_gp                                                                % #points of the GP ODE enforcement grid
        niterGP                                                             % #iterations for generalized profiling
        tolGP                                                               % convergence tolerance for generalized profiling
        icweightGP                                                          % GP hidden state initial condition anchor weight
        scaleGP                                                             % logical for per-state GP residual scaling
        initGP                                                              % GP hidden state initialization mode
        ninnerGP                                                            % #inner Gauss-Newton steps for profiling
        hiddenknotsGP                                                       % B-spline knots for unobserved states
        nestedGP                                                            % logical for the nested GP scheme

        levelMAGI                                                           % MAGI discretization level
        niterMAGI                                                           % #HMC iterations
        nstepsMAGI                                                          % #leapfrog steps per HMC iteration
        burninMAGI                                                          % HMC burn-in proportion
        bandMAGI                                                            % band matrix approximation size
        stepsizeMAGI                                                        % initial leapfrog step size factor
        kernelMAGI                                                          % GP covariance kernel
        sigmaMAGI                                                           % known measurement error SDs
        fixedsigmaMAGI                                                      % logical for treating sigma as known
        summaryMAGI                                                         % posterior summary ('mean' or 'median')
        initMAGI                                                            % MAGI trajectory initialization mode
        verboseMAGI                                                         % logical for MagiSolver progress messages

        M_pc                                                                % #points of the PC enforcement grid
        niterPC                                                             % #outer iterations for the cascade
        tolPC                                                               % convergence tolerance for the cascade
        ninnerPC                                                            % #inner Gauss-Newton steps
        lambdaPC                                                            % per-equation ODE penalty weight
        lambdadataPC                                                        % data fidelity weight
        lambdaicPC                                                          % initial condition penalty weight
        lambdapositivityPC                                                  % positivity barrier weight
        anchorallPC                                                         % logical for anchoring observed states too
        initPC                                                              % PC hidden state initialization mode
        nstartPC                                                            % #starting points for the outer problem
        solverPC                                                            % 'fmincon' or 'builtin'

        init                                                                % vector of initial conditions
        P_CV                                                                % initial condition prior CV
        sigma                                                               % Kalman filter process noise
        Q_mult                                                              % Kalman filter Q matrix scale
        prior                                                               % model parameter prior (struct)
        
        settings_smoothing                                                  % structs containing default/user preferences
        settings_filter
        settings_inference
        settings_gp
        settings_magi
        settings_pc
        output                                                              % inference output struct
        
        smoother                                                            % internal smoother object
        filter                                                              % internal EKF object
        gm                                                                  % internal gradient matching object
        tm                                                                  % internal trajectory matching object
        gp                                                                  % internal generalized profiling object
        magi                                                                % internal MAGI object
        pc                                                                  % internal parameter cascade object
        
        beta                                                                % model parameter vector
        filtered                                                            % Kalman filtered/smoothed states
    end
    
    
    methods
        %% Constructor -----------------------------------------------------
        function obj = EGM(system, data, varargin)
            [system, data] = obj.parse_initial(system, data, varargin{:});
            
            default_t = 0:size(data, 1)-1;                                  % default arguments for parser
            default_observed = 1:size(data, 2);
            
            initial = system.k0';
            default_AutoKnots = true;
            default_Knots = repmat({linspace(data.t(1), data.t(end), round((data.T-1)/2)+1)}, ...
                                   1, length(data.observed));
            default_InteractiveSmoothing = false;
            
            default_LB = .25 * initial;
            default_UB = 4 .* initial + .0001 * mean(initial);
            default_PositiveStates = true;
            default_TimePoints = data.t;
            
            default_MaxIterationsSM = 20;
            default_ConvergenceTolSM = 1e-3;
            default_MaxIterationsGM = 20;
            default_ConvergenceTolGM = 1e-4;

            default_GridSizeGP = 201;                                       % generalized profiling defaults
            default_MaxIterationsGP = 20;
            default_ConvergenceTolGP = 1e-4;
            default_ICWeightGP = .1;
            default_ScaleModelGP = true;
            default_InitializationGP = 'integrate';
            default_InnerIterationsGP = 8;
            default_HiddenKnotsGP = [];
            default_NestedGP = [];                                          % empty: nested iff states are unobserved

            default_DiscretizationMAGI = 1;                                 % MAGI defaults
            default_IterationsMAGI = 100;
            default_StepsMAGI = 200;
            default_BurninMAGI = .25;
            default_BandSizeMAGI = 20;
            default_StepSizeMAGI = [];
            default_KernelMAGI = 'generalMatern';
            default_SigmaMAGI = [];
            default_FixedSigmaMAGI = false;
            default_SummaryMAGI = 'mean';
            default_InitializationMAGI = 'magi';
            default_VerboseMAGI = false;

            default_GridSizePC = 201;                                       % parameter cascade defaults
            default_MaxIterationsPC = 20;
            default_ConvergenceTolPC = 1e-4;
            default_InnerIterationsPC = 20;
            default_LambdaPC = 1e2;
            default_LambdaDataPC = 1;
            default_LambdaICPC = 1e-2;
            default_LambdaPositivityPC = 1e3;
            default_AnchorAllPC = false;
            default_InitializationPC = 'integrate';
            default_MultiStartPC = 5;
            default_SolverPC = 'fmincon';

            default_InitialConditions = system.x0';
            default_PCV = .25;
            default_Sigma = .05;
            default_QMult = 1;
            % default_PCV = .5;
            % default_Sigma = .1;
            % default_QMult = 1;

            default_Prior = struct('mean', 0, 'prec', 0);
            
            parser = inputParser;
            parser.KeepUnmatched = true;                                    % add all arguments to parser
            addRequired(parser, 'system', @(x) isa(x, 'System') || isstring(string(x)) && numel(string(x)) == 1);
            addRequired(parser, 'data', @(x) isstruct(x) || isnumeric(x) && ndims(x) == 3 && size(x, 1) > 1);
            addOptional(parser, 't', default_t, @(x) isnumeric(x) && numel(unique(x)) == data.T);
            addOptional(parser, 'observed', default_observed, @(x) isnumeric(x) && numel(unique(x)) == data.L);
            
            addParameter(parser, 'AutoKnots', default_AutoKnots, @islogical);
            addParameter(parser, 'Knots', default_Knots, @(x) (iscell(x) && length(x) == data.L ...
                                                               && all(cellfun(@isnumeric, x))) ...
                                                           || isnumeric(x));
            addParameter(parser, 'InteractiveSmoothing', default_InteractiveSmoothing, @islogical);

            addParameter(parser, 'LB', default_LB, @(x) all(x < initial));
            addParameter(parser, 'UB', default_UB, @(x) all(x > initial));
            addParameter(parser, 'PositiveStates', default_PositiveStates, @islogical);
            addParameter(parser, 'TimePoints', default_TimePoints, @(x) all(data.t(1) <= x & x <= data.t(end)));
            
            addParameter(parser, 'MaxIterationsSM', default_MaxIterationsSM, @isscalar);
            addParameter(parser, 'ConvergenceTolSM', default_ConvergenceTolSM, @isscalar);
            addParameter(parser, 'MaxIterationsGM', default_MaxIterationsGM, @isscalar);
            addParameter(parser, 'ConvergenceTolGM', default_ConvergenceTolGM, @isscalar);

            addParameter(parser, 'GridSizeGP', default_GridSizeGP, @(x) isscalar(x) && x >= 11);
            addParameter(parser, 'MaxIterationsGP', default_MaxIterationsGP, @isscalar);
            addParameter(parser, 'ConvergenceTolGP', default_ConvergenceTolGP, @isscalar);
            addParameter(parser, 'ICWeightGP', default_ICWeightGP, @(x) isscalar(x) && x > 0);
            addParameter(parser, 'ScaleModelGP', default_ScaleModelGP, @islogical);
            addParameter(parser, 'InitializationGP', default_InitializationGP, ...
                         @(x) any(strcmpi(char(x), {'integrate', 'constant'})));
            addParameter(parser, 'InnerIterationsGP', default_InnerIterationsGP, @isscalar);
            addParameter(parser, 'HiddenKnotsGP', default_HiddenKnotsGP, @(x) isnumeric(x) || iscell(x));
            addParameter(parser, 'NestedGP', default_NestedGP, @(x) isempty(x) || islogical(x));

            addParameter(parser, 'DiscretizationMAGI', default_DiscretizationMAGI, @(x) isscalar(x) && x >= 0);
            addParameter(parser, 'IterationsMAGI', default_IterationsMAGI, @isscalar);
            addParameter(parser, 'StepsMAGI', default_StepsMAGI, @isscalar);
            addParameter(parser, 'BurninMAGI', default_BurninMAGI, @(x) isscalar(x) && x >= 0 && x < 1);
            addParameter(parser, 'BandSizeMAGI', default_BandSizeMAGI, @isscalar);
            addParameter(parser, 'StepSizeMAGI', default_StepSizeMAGI, @isnumeric);
            addParameter(parser, 'KernelMAGI', default_KernelMAGI, @(x) ischar(x) || isstring(x));
            addParameter(parser, 'SigmaMAGI', default_SigmaMAGI, @isnumeric);
            addParameter(parser, 'FixedSigmaMAGI', default_FixedSigmaMAGI, @islogical);
            addParameter(parser, 'SummaryMAGI', default_SummaryMAGI, @(x) any(strcmpi(char(x), {'mean', 'median'})));
            addParameter(parser, 'InitializationMAGI', default_InitializationMAGI, ...
                         @(x) any(strcmpi(char(x), {'magi', 'integrate'})));
            addParameter(parser, 'VerboseMAGI', default_VerboseMAGI, @islogical);

            addParameter(parser, 'GridSizePC', default_GridSizePC, @(x) isscalar(x) && x >= 11);
            addParameter(parser, 'MaxIterationsPC', default_MaxIterationsPC, @isscalar);
            addParameter(parser, 'ConvergenceTolPC', default_ConvergenceTolPC, @isscalar);
            addParameter(parser, 'InnerIterationsPC', default_InnerIterationsPC, @isscalar);
            addParameter(parser, 'LambdaPC', default_LambdaPC, @(x) isnumeric(x) && all(x > 0));
            addParameter(parser, 'LambdaDataPC', default_LambdaDataPC, @(x) isscalar(x) && x > 0);
            addParameter(parser, 'LambdaICPC', default_LambdaICPC, @(x) isscalar(x) && x >= 0);
            addParameter(parser, 'LambdaPositivityPC', default_LambdaPositivityPC, @(x) isscalar(x) && x >= 0);
            addParameter(parser, 'AnchorAllPC', default_AnchorAllPC, @islogical);
            addParameter(parser, 'InitializationPC', default_InitializationPC, ...
                         @(x) any(strcmpi(char(x), {'integrate', 'constant'})));
            addParameter(parser, 'MultiStartPC', default_MultiStartPC, @(x) isscalar(x) && x >= 1);
            addParameter(parser, 'SolverPC', default_SolverPC, @(x) any(strcmpi(char(x), {'fmincon', 'builtin'})));

            addParameter(parser, 'InitialConditions', default_InitialConditions, @(x) numel(x) == system.K);
            addParameter(parser, 'PCV', default_PCV, @(x) isscalar(x) && x > 0);
            addParameter(parser, 'Sigma', default_Sigma, @(x) isscalar(x) && x > 0);
            addParameter(parser, 'QMult', default_QMult, @(x) isscalar(x) && x > 0);
            addParameter(parser, 'Prior', default_Prior, @(x) isfield(x, 'mean') && numel(x.mean) == system.P ...
                                                           && (isfield(x, 'prec') && all(size(x.prec) == system.P) ...
                                                               && issymmetric(x.prec) && all(eig(x.prec) >= 0) ...
                                                            || isfield(x, 'sd') && numel(x.sd) == system.P ...
                                                               && all(x.sd > 0) ...
                                                            || isfield(x, 'cv') && numel(x.cv) == system.P ...
                                                               && all(x.cv > 0)));
            parse(parser, system, data, varargin{:});                       % initial parsing of system and data
            [obj.system, obj.data] = obj.parse_initial(parser.Results.system, parser.Results.data, varargin{:});
            obj.parse_parameters(parser);
            
            obj.data.T_fine = 81;                                           % fixed grid size for smooth model fits
            obj.data.t_fine = linspace(obj.data.t(1), obj.data.t(end), obj.data.T_fine);
            
            weights = ones(length(obj.t_gm), obj.system.K);                 % standard gradient matching time point weights 
            weights([1 end], :) = 0;                                        %   to omit interval ends (with low spline accuracy)
            
                                                                            % save settings for each main component
            obj.settings_smoothing = struct('order', 4, 'autoknots', obj.autoknots, 'knots', {obj.knots}, ...
                                            'positive', obj.positive, 't_gm', obj.t_gm, 'niter', obj.niterSM, ...
                                            'tol', obj.tolSM, 'interactive', obj.interactive);
            obj.settings_filter = struct('sigma', obj.sigma, 'Q_mult', obj.Q_mult, 'P_CV', obj.P_CV, 'x_init', obj.init, ...
                                         't_gm', obj.t_gm);
            obj.settings_inference = struct('lb', obj.lb, 'ub', obj.ub, 'positive', obj.positive, 't_gm', obj.t_gm, ...
                                            'weights', weights, 'niter', obj.niterGM, 'tol', obj.tolGM, 'prior', obj.prior);

            knots_gp = cell(1, obj.system.K);                               % generalized profiling splines every state, so
            hidden_knots = unique([obj.knots{:}]);                          % hidden states need knots as well. Their
            if ~isempty(obj.hiddenknotsGP)                                  % flexibility matters: too stiff and they bias
                hidden_knots = reshape(obj.hiddenknotsGP, 1, []);           % beta, too flexible and they absorb the
                hidden_knots = max(obj.data.t(1), min(obj.data.t(end), hidden_knots));
                hidden_knots = sort(unique([obj.data.t(1) hidden_knots obj.data.t(end)]));
                hidden_knots = (hidden_knots - obj.data.t(1)) / range(obj.data.t);
            end                                                             % information identifying it, so the default is
            for k = 1:obj.system.K                                          % the union of the observed knot sets and
                l = find(obj.data.observed == k, 1);                        % HiddenKnotsGP overrides it
                if isempty(l), knots_gp{k} = hidden_knots; else, knots_gp{k} = obj.knots{l}; end
            end

            obj.settings_gp = struct('order', 4, 'knots', {knots_gp}, 'positive', obj.positive, ...
                                     'lb', obj.lb, 'ub', obj.ub, 'prior', obj.prior, 'M', obj.M_gp, ...
                                     'niter', obj.niterGP, 'tol', obj.tolGP, 'ic_weight', obj.icweightGP, ...
                                     'scale_model', obj.scaleGP, 'init_mode', obj.initGP, ...
                                     'ninner', obj.ninnerGP);
            if ~isempty(obj.nestedGP), obj.settings_gp.nested = obj.nestedGP; end

            obj.settings_magi = struct('level', obj.levelMAGI, 'niter', obj.niterMAGI, ...
                                       'nsteps', obj.nstepsMAGI, 'burnin', obj.burninMAGI, ...
                                       'band', obj.bandMAGI, 'stepsize', obj.stepsizeMAGI, ...
                                       'kernel', obj.kernelMAGI, 'sigma', obj.sigmaMAGI, ...
                                       'fixed_sigma', obj.fixedsigmaMAGI, 'verbose', obj.verboseMAGI, ...
                                       'summary', obj.summaryMAGI, 'init_mode', obj.initMAGI, ...
                                       'positive', obj.positive, 'lb', obj.lb, 'ub', obj.ub);

            obj.settings_pc = struct('order', 4, 'knots', {knots_gp}, 'positive', obj.positive, ...
                                     'lb', obj.lb, 'ub', obj.ub, 'prior', obj.prior, 'M', obj.M_pc, ...
                                     'niter', obj.niterPC, 'tol', obj.tolPC, 'ninner', obj.ninnerPC, ...
                                     'inner_tol', 1e-8, 'fd_step', 1e-6, 'nsearch', 8, 'nstart', obj.nstartPC, 'solver', obj.solverPC, 'lambda', obj.lambdaPC, ...
                                     'lambda_data', obj.lambdadataPC, 'lambda_ic', obj.lambdaicPC, ...
                                     'lambda_positivity', obj.lambdapositivityPC, ...
                                     'anchor_all', obj.anchorallPC, 'init_mode', obj.initPC);
        end

                                                                        % Basic parsing to allow conditional input validations
        function [system, data] = parse_initial(~, system, data, varargin)  
            if ~isa(system, 'System')                                       % process model file if provided
                namevalue = arrayfun(@(idx) iscellstr(varargin(idx)) || isstring(varargin{idx}), 1:length(varargin));
                first_namevalue = find(namevalue, 1);
                if isempty(first_namevalue), first_namevalue = length(varargin)+1; end
                system = System(string(system), varargin{first_namevalue:end});
            end
            if isstruct(data)                                               % default any missing fields
                if ~isfield(data, 'traces') && isfield(data, 'y'), data.traces = data.y; end
                if ~isfield(data, 't'), data.t = 0:size(data.traces, 1)-1; end
                if ~isfield(data, 'observed'), data.observed = 1:size(data.traces, 2); end
                if ~isfield(data, 'init'), data.init = system.x0' + 1e-4; end
            else                                                            % components provided separately
                traces = data;                                              % array with measurements instead of struct
                if ~iscellstr(varargin(1)) && ~isstring(varargin{1})        % recursively check if optional or Name/Value
                    t = sort(unique(reshape(varargin{1}, 1, [])));
                    if ~iscellstr(varargin(2)) && ~isstring(varargin{2})
                        observed = sort(unique(reshape(varargin{2}, 1, [])));
                    else
                        observed = 1:size(traces, 2);
                    end
                else
                    t = 0:size(traces, 1)-1;
                    observed = 1:size(traces, 2);
                end                                                         % compile into struct
                data = struct('traces', traces, 't', t, 'observed', observed, 'init', ones(1, system.K));
            end
            [data.T, data.L, data.N] = size(data.traces);                   % include dimensions for notational convenience
        end


        function parse_parameters(obj, parser)                          % Parse Name/Value constructor arguments
            obj.autoknots = parser.Results.AutoKnots;
            obj.knots = cell(1, obj.data.L);
            parsed_knots = parser.Results.Knots;
            if ~iscell(parsed_knots)                                        % duplicate knots for each observed state
                parsed_knots = repmat({parsed_knots}, 1, obj.data.L);
            end
            for state = 1:length(obj.data.observed)                         % clean up and sort knots 
                truncated = max(obj.data.t(1), min(obj.data.t(end), parsed_knots{state}));
                arranged = sort(unique([obj.data.t(1) reshape(truncated, 1, []) obj.data.t(end)]));
                obj.knots{state} = (arranged - obj.data.t(1)) / range(obj.data.t);
            end
            obj.autoknots = parser.Results.AutoKnots && ismember("Knots", string(parser.UsingDefaults));
            obj.interactive = parser.Results.InteractiveSmoothing;
            
            obj.lb = parser.Results.LB;
            obj.ub = parser.Results.UB;
            obj.positive = parser.Results.PositiveStates;
            obj.t_gm = sort(unique(parser.Results.TimePoints));

            obj.niterSM = max(1, round(parser.Results.MaxIterationsSM));    % include minima
            obj.tolSM = max(1e-12, parser.Results.ConvergenceTolSM);
            obj.niterGM = max(1, round(parser.Results.MaxIterationsGM));
            obj.tolGM = max(1e-12, parser.Results.ConvergenceTolGM);

            obj.M_gp = max(11, round(parser.Results.GridSizeGP));
            obj.niterGP = max(1, round(parser.Results.MaxIterationsGP));
            obj.tolGP = max(1e-12, parser.Results.ConvergenceTolGP);
            obj.icweightGP = parser.Results.ICWeightGP;
            obj.scaleGP = parser.Results.ScaleModelGP;
            obj.initGP = lower(char(parser.Results.InitializationGP));
            obj.ninnerGP = max(1, round(parser.Results.InnerIterationsGP));
            obj.hiddenknotsGP = parser.Results.HiddenKnotsGP;
            obj.nestedGP = parser.Results.NestedGP;

            obj.levelMAGI = max(0, round(parser.Results.DiscretizationMAGI));
            obj.niterMAGI = max(2, round(parser.Results.IterationsMAGI));
            obj.nstepsMAGI = max(1, round(parser.Results.StepsMAGI));
            obj.burninMAGI = parser.Results.BurninMAGI;
            obj.bandMAGI = max(1, round(parser.Results.BandSizeMAGI));
            obj.stepsizeMAGI = parser.Results.StepSizeMAGI;
            obj.kernelMAGI = char(parser.Results.KernelMAGI);
            obj.sigmaMAGI = parser.Results.SigmaMAGI;
            obj.fixedsigmaMAGI = parser.Results.FixedSigmaMAGI;
            obj.summaryMAGI = lower(char(parser.Results.SummaryMAGI));
            obj.initMAGI = lower(char(parser.Results.InitializationMAGI));
            obj.verboseMAGI = parser.Results.VerboseMAGI;

            obj.M_pc = max(11, round(parser.Results.GridSizePC));
            obj.niterPC = max(1, round(parser.Results.MaxIterationsPC));
            obj.tolPC = max(1e-12, parser.Results.ConvergenceTolPC);
            obj.ninnerPC = max(1, round(parser.Results.InnerIterationsPC));
            obj.lambdaPC = parser.Results.LambdaPC;
            obj.lambdadataPC = parser.Results.LambdaDataPC;
            obj.lambdaicPC = parser.Results.LambdaICPC;
            obj.lambdapositivityPC = parser.Results.LambdaPositivityPC;
            obj.anchorallPC = parser.Results.AnchorAllPC;
            obj.initPC = lower(char(parser.Results.InitializationPC));
            obj.nstartPC = max(1, round(parser.Results.MultiStartPC));
            obj.solverPC = lower(char(parser.Results.SolverPC));

            obj.init = reshape(parser.Results.InitialConditions, 1, []);
            obj.P_CV = parser.Results.PCV;
            obj.sigma = parser.Results.Sigma;
            obj.Q_mult = parser.Results.QMult;

            obj.prior = parser.Results.Prior;
            obj.prior.mean = flatten(obj.prior.mean);
            if isfield(obj.prior, 'cv')                                     % infer prior variability depending on input
                obj.prior.cv = flatten(obj.prior.cv);
                obj.prior.mean(obj.prior.mean == 0 & isinf(obj.prior.cv)) = 1;
                obj.prior.sd = obj.prior.cv .* obj.prior.mean;
            end
            if isfield(obj.prior, 'sd')
                obj.prior.sd = flatten(obj.prior.sd);
                warning('off','MATLAB:singularMatrix')
                obj.prior.prec = inv(diag(obj.prior.sd.^2));
                warning('on','MATLAB:singularMatrix')
            end
            if ~isfield(obj.prior, 'mult'), obj.prior.mult = 1; end
        end
        
        
        %% Estimation ------------------------------------------------------
        function out = estimate(obj, beta_init, ~)                      % Estimate model parameters (using EGM)
            ws = warning('error', 'MATLAB:nearlySingularMatrix');           % turn off nearly singular matrix warnings
            silent = nargin == 3;
                                                                            % B-spline smoothing of observations
            obj.smoother = Smoother(obj.system, obj.data, obj.settings_smoothing);  % (Smoother object from GMGTS implementation)
            obj.output = obj.smoother.smooth();
            if ~silent, toc_sm = toc, else, toc_sm = toc; end     %#ok<NOPRT>

            indices = ismember(obj.data.t, obj.t_gm);                       % subsample time points and measurements if applicable
            obj.output.traces = obj.output.traces(indices, :);
            obj.output.t_data = obj.t_gm;
            for k = 1:obj.data.L                                            % apply subsampling to B-spline smoothing
                obj.output.basis{k} = obj.output.basis{k}(indices, :);
                obj.output.dbasis{k} = obj.output.dbasis{k}(indices, :);
            end
                                                                            % instantiate filter (EKS) and gradient matching components
            obj.filter = Filter(obj.system, obj.output, obj.settings_filter);
            obj.gm = GM(obj.system, obj.output, obj.settings_inference);
            
            obj.beta = zeros(obj.settings_inference.niter+1, obj.system.P); % allocate model parameter estimate iterations
            
            tic
            for iter = 1:obj.settings_inference.niter                       % iterate model parameter estimates
                if ~silent, fprintf('%d ', iter), end                       % optionally display iteration number
                                                                            % optimize initial condition prior mean
                if iter == 1, obj.beta(1, :) = obj.init_ICmean(beta_init); end

                obj.filtered = obj.filter.EKS(obj.beta(iter, :));           % filter and gradient matching

                                                                            % extend EKS covariance matrices with original time points
                                                                            % (to correctly approximate covariances with B-spline smoothing)
                obj.filtered.covar = obj.augment_covariances(obj.filtered.covar, obj.output.variances_sm);

                                                                            % gradient matching step
                obj.beta(iter+1, :) = obj.gm.estimate(obj.beta(iter, :), obj.filtered);
                
                                                                            % optionally current relative step size
                if ~mod(iter, 10) && ~silent, fprintf('(%.3e)\n', eucl_rel(obj.beta(iter, :), obj.beta(iter+1, :))), end
                
                if iter == obj.settings_inference.niter || ...              % break upon reaching step tolerance
                   eucl_rel(obj.beta(iter, :), obj.beta(iter+1, :)) < obj.settings_inference.tol
                    obj.beta = obj.beta(1:iter+1, :);
                    fprintf('(%.3e)\n', eucl_rel(obj.beta(iter, :), obj.beta(iter+1, :)))
                    break
                end
            end
            fprintf('\n')
                                                                            % optionally display computation time
            if ~silent, toc_est = toc, else, toc_est = toc; end     %#ok<NOPRT>
            if ~silent, fprintf('total time: %.3f seconds\n', toc_sm + toc_est), end

            data_updated = obj.data;                                        % update initial condition mean from EKS
            data_updated.init = obj.filtered.xbar(:, 1)';
            fitted = obj.system.integrate(obj.beta(end, :), data_updated);  % integrate system and compute rhs
            fitted_fine = obj.system.integrate(obj.beta(end, :), data_updated, obj.data.t_fine);
            dfitted = obj.system.rhs(fitted, obj.data.t, obj.beta(end, :));
            dfitted_fine = obj.system.rhs(fitted_fine, obj.data.t_fine, obj.beta(end, :));

            if obj.positive
                fitted = max(1e-12, fitted);                                % force positive predicted states
                fitted_fine = max(1e-12, fitted_fine);
            end
            
            obj.output.init = data_updated.init;                            % save results in output
            obj.output.fitted = fitted;
            obj.output.dfitted = dfitted;
            obj.output.fitted_fine = fitted_fine;
            obj.output.dfitted_fine = dfitted_fine;
            
            obj.output.time = [toc_sm toc_est];
            obj.output.beta = obj.beta(end, :);
            obj.output.filtered = obj.filtered;
            out = obj.output;
            
            warning(ws);                                                    % reset nearly singular matrix warnings
        end


        function beta_opt = init_ICmean(obj, beta_init)                 % Crude initial condition prior mean priming
            center = obj.settings_filter.x_init;                            % start from supplied initial condition guess
            center(obj.data.observed) = obj.output.smoothed(1, :);          % estimate observed states from B-spline smoothing
            multipliers = 4.^(-1:.5:1);                                     % consider multiplies of the starting point
            uncertain = find(~ismember(1:obj.system.K, obj.data.observed)); % default: scan for each hidden states
            switch string(obj.system.name)                                  % specific settings for certain systems
                case "STAT5"
                    uncertain = [3 4];
                case "repressilator"
                    uncertain = [4 7];
                case "generalizedLV"
                    uncertain = 1;
                case "bifunctional_TCS"
                    uncertain = 4;
                case "ERK_MAPK"                                         % scan unknown enzyme levels KK(0), P(0)
                    uncertain = [4 5];
            end
                                                                            % all combinations of initial conditions to consider
            conditions = repmat(center', [1 length(multipliers) * ones(1, length(uncertain))]);
            switch length(uncertain)
                case 1
                    conditions(uncertain, :) = center(uncertain) * multipliers;
                    conditions = conditions';
                    if string(obj.system.name) == "generalizedLV"           % consider equal initial conditions for states 1,2
                        conditions(:, 2) = conditions(:, 1);
                    end
                case 2                                                      % square number of combinations
                    mult = ones(size(conditions));                          % (for more uncertain ICs, extend switch statement or sample randomly)        
                    mult(uncertain(1), :, :) = repmat(multipliers, 1, 1, length(multipliers));
                    mult(uncertain(2), :, :) = repmat(reshape(multipliers, 1, 1, []), 1, length(multipliers), 1);
                    conditions = reshape(conditions .* mult, length(center), [])';
                    if string(obj.system.name) == "repressilator"           % consider equal initial conditions for states 4-6 and 7-9  
                        conditions(:, [5 6]) = repmat(conditions(:, 4), 1, 2);
                        conditions(:, [8 9]) = repmat(conditions(:, 7), 1, 2);
                    end
            end

            if string(obj.system.name) == "STAT5"                     % additional step to satisfy inequality constraint x3(0) > x4(0)
                conditions(:, 4) = min(conditions(:, 3), conditions(:, 4));
            end

            errors = inf(1, size(conditions, 1));                          % allocate parameter estimates and generalized 
            betas = zeros(size(conditions, 1), obj.system.P);               % sums of squares for each vector of initial conditions

            for cond = 1:size(conditions, 1)                                % one GM step per tested IC vector
                try                                                         % skip IC vectors for which filtering fails
                    obj.filter.update_ICmean(conditions(cond, :));          % (e.g. CVODE errors), leaving error at Inf
    
                    obj.filtered = obj.filter.EKS(beta_init);               % filter and gradient matching
                    obj.filtered.covar = obj.augment_covariances(obj.filtered.covar, obj.output.variances_sm);
                    [betas(cond, :), errors(cond), ~] = obj.gm.estimate(beta_init, obj.filtered);
                catch ME
                    fprintf('IC scan: skipping condition %d/%d (%s)\n', cond, size(conditions, 1), ME.message)
                end
            end

            if all(isinf(errors))                                           % no usable IC vector at all
                error('EGM:ICscanFailed', 'Initial condition scan failed for all %d candidate IC vectors.', ...
                      size(conditions, 1))
            end

            [~, cond] = min(errors(:));                                     % pick IC vector with smallest associated sum of squares
            xinit_opt = conditions(cond, :);
            beta_opt = betas(cond, :);
            obj.filter.update_ICmean(xinit_opt);                            % save updated parameter estimate and IC vector
        end

                                                                        % Extend EKS covariance matrices with original time points
        function augmented = augment_covariances(obj, covar, variances_sm)
            K = obj.system.K;                                               % abbreviate numbers of states and time points
            L = obj.data.L;
            T = length(obj.data.t);
            T_gm = length(obj.t_gm);
            subsample = find(ismember(obj.data.t, obj.t_gm));               % time subsampling indices
            subsamples = reshape(subsample' + T*(0:L-1), 1, []);

            augmented = zeros(K*T_gm + L*T, K*T_gm + L*T);                  % incorporate original data time points into EKS covariance matrix
            augmented(1:K*T_gm, 1:K*T_gm) = covar(1:K*T_gm, 1:K*T_gm);      % nonzero covariances only at corresponding elements of covar
            augmented(1:K*T_gm, K*T_gm + subsamples) = covar(1:K*T_gm, K*T_gm+1:end);
            augmented(K*T_gm + subsamples, 1:K*T_gm) = covar(K*T_gm+1:end, 1:K*T_gm);
            augmented(K*T_gm+1:end, K*T_gm+1:end) = diag(variances_sm(:));  % exception: estimated measurement error variances
        end


        function out = estimate_TM(obj, beta_init, x_init)              % Estimate using trajectory matching
            ws = warning('error', 'MATLAB:nearlySingularMatrix');
            silent = nargin == 3;
            
            x_init(obj.data.observed) = obj.data.traces(1, :);              % trajectory matching component
            obj.tm = TM(obj.data, obj.system, obj.settings_inference, beta_init, x_init);
            
            tic
            out = obj.tm.optimize(true);                                        % optimize initial conditions as well
            if silent, toc_est = toc; else, toc_est = toc, end

            obj.beta = out.beta_fs;                                         % update using optimized initial conditions
            obj.output.init = out.init;
            int_data = obj.data;
            int_data.init = obj.output.init;

            fitted = obj.system.integrate(obj.beta, int_data);              % integrate system and compute rhs
            fitted_fine = obj.system.integrate(obj.beta, int_data, obj.data.t_fine);    % analogous fine-grained model fits
            dfitted = obj.system.rhs(fitted, obj.data.t, obj.beta);
            dfitted_fine = obj.system.rhs(fitted_fine, obj.data.t_fine, obj.beta);

            if obj.positive
                fitted = max(1e-12, fitted);                                % force positive predicted states
                fitted_fine = max(1e-12, fitted_fine);
            end
            
            obj.output.fitted = fitted;                                     % save results in output
            obj.output.dfitted = dfitted;
            obj.output.fitted_fine = fitted_fine;
            obj.output.dfitted_fine = dfitted_fine;

            obj.output.time = [0 toc_est];
            obj.output.beta = obj.beta;
            out = obj.output;
            
            warning(ws);
        end


        function out = estimate_GP(obj, beta_init, x_init)              % Estimate using generalized profiling
            ws = warning('error', 'MATLAB:nearlySingularMatrix');
            silent = nargin == 3;

            if nargin < 3 || isempty(x_init), x_init = obj.init; end        % default to the supplied IC guess
            x_init = reshape(x_init, 1, []);
            x_init(obj.data.observed) = obj.data.traces(1, :, 1);

                                                                            % generalized profiling component
            obj.gp = GP(obj.data, obj.system, obj.settings_gp, beta_init, x_init);

            tic
            out = obj.gp.optimize();                                        % profile splines and parameters jointly
            if silent, toc_est = toc; else, toc_est = toc, end   %#ok<NOPRT>

            obj.beta = out.beta_gp;                                         % initial conditions follow from the splines
            obj.output = out;
            int_data = obj.data;
            int_data.init = obj.output.init;

            try                                                             % integrate system and compute rhs
                fitted = obj.system.integrate(obj.beta, int_data);          % (for comparison only: estimation itself
                fitted_fine = obj.system.integrate(obj.beta, int_data, obj.data.t_fine);    % never solves the ODE)
                dfitted = obj.system.rhs(fitted, obj.data.t, obj.beta);
                dfitted_fine = obj.system.rhs(fitted_fine, obj.data.t_fine, obj.beta);
            catch ME
                disp(ME)                                                    % fall back on the profiled splines
                fitted = out.smoothed_gp;
                fitted_fine = out.smoothed_fine;
                dfitted = out.dsmoothed_gp;
                dfitted_fine = out.dsmoothed_fine;
            end

            if obj.positive
                fitted = max(1e-12, fitted);                                % force positive predicted states
                fitted_fine = max(1e-12, fitted_fine);
            end

            obj.output.fitted = fitted;                                     % save results in output
            obj.output.dfitted = dfitted;
            obj.output.fitted_fine = fitted_fine;
            obj.output.dfitted_fine = dfitted_fine;

            obj.output.time = [0 toc_est];
            obj.output.beta = obj.beta;
            out = obj.output;

            warning(ws);
        end


        function out = estimate_MAGI(obj, beta_init, x_init)            % Estimate using manifold-constrained GP inference
            silent = nargin == 3;

            if nargin < 3 || isempty(x_init), x_init = obj.init; end        % default to the supplied IC guess
            x_init = reshape(x_init, 1, []);
            x_init(obj.data.observed) = obj.data.traces(1, :, 1);

                                                                            % MAGI component (wraps the authors' sampler)
            obj.magi = MAGI(obj.data, obj.system, obj.settings_magi, beta_init, x_init);

            tic
            out = obj.magi.optimize();                                      % posterior over trajectories and parameters
            if silent, toc_est = toc; else, toc_est = toc, end   %#ok<NOPRT>

            obj.beta = out.beta_magi;                                       % initial conditions from the posterior mean
            obj.output = out;
            int_data = obj.data;
            int_data.init = obj.output.init;

            try                                                             % integrate for comparability with the other
                fitted = obj.system.integrate(obj.beta, int_data);          % estimators; MAGI itself needs no ODE solve
                fitted_fine = obj.system.integrate(obj.beta, int_data, obj.data.t_fine);
                dfitted = obj.system.rhs(fitted, obj.data.t, obj.beta);
                dfitted_fine = obj.system.rhs(fitted_fine, obj.data.t_fine, obj.beta);
            catch ME
                disp(ME)                                                    % fall back on the sampled trajectories
                fitted = interp1(out.tvec_magi', out.smoothed_magi, obj.data.t');
                fitted_fine = interp1(out.tvec_magi', out.smoothed_magi, obj.data.t_fine');
                dfitted = obj.system.rhs(fitted, obj.data.t, obj.beta);
                dfitted_fine = obj.system.rhs(fitted_fine, obj.data.t_fine, obj.beta);
            end

            if obj.positive
                fitted = max(1e-12, fitted);                                % force positive predicted states
                fitted_fine = max(1e-12, fitted_fine);
            end

            obj.output.fitted = fitted;                                     % save results in output
            obj.output.dfitted = dfitted;
            obj.output.fitted_fine = fitted_fine;
            obj.output.dfitted_fine = dfitted_fine;

            obj.output.time = [0 toc_est];
            obj.output.beta = obj.beta;
            out = obj.output;
        end


        function out = estimate_PC(obj, beta_init, x_init)              % Estimate using the Ramsay parameter cascade
            ws = warning('error', 'MATLAB:nearlySingularMatrix');
            silent = nargin == 3;

            if nargin < 3 || isempty(x_init), x_init = obj.init; end
            x_init = reshape(x_init, 1, []);
            x_init(obj.data.observed) = obj.data.traces(1, :, 1);

                                                                            % parameter cascade component
            obj.pc = PC(obj.data, obj.system, obj.settings_pc, beta_init, x_init);

            tic
            out = obj.pc.optimize();                                        % nested inner/outer optimization
            if silent, toc_est = toc; else, toc_est = toc, end   %#ok<NOPRT>

            obj.beta = out.beta_pc;
            obj.output = out;
            int_data = obj.data;
            int_data.init = obj.output.init;

            try                                                             % integrate for comparability with the other
                fitted = obj.system.integrate(obj.beta, int_data);          % estimators; the cascade never integrates
                fitted_fine = obj.system.integrate(obj.beta, int_data, obj.data.t_fine);
                dfitted = obj.system.rhs(fitted, obj.data.t, obj.beta);
                dfitted_fine = obj.system.rhs(fitted_fine, obj.data.t_fine, obj.beta);
            catch ME
                disp(ME)                                                    % fall back on the profiled splines
                fitted = out.smoothed_pc;
                fitted_fine = out.smoothed_fine;
                dfitted = out.dsmoothed_pc;
                dfitted_fine = out.dsmoothed_fine;
            end

            if obj.positive
                fitted = max(1e-12, fitted);
                fitted_fine = max(1e-12, fitted_fine);
            end

            obj.output.fitted = fitted;
            obj.output.dfitted = dfitted;
            obj.output.fitted_fine = fitted_fine;
            obj.output.dfitted_fine = dfitted_fine;

            obj.output.time = [0 toc_est];
            obj.output.beta = obj.beta;
            out = obj.output;

            warning(ws);
        end
    end
end