classdef GP < handle                                                    % Generalized profiling estimator
%GP implements generalized profiling (parameter cascading) for ODE model
%parameter estimation, following Simpson, Bennett, Johnston and Baker,
%"Parameter estimation for differential equation models using generalized
%profiling: a computational tutorial" (arXiv:2602.04891), and the reference
%Julia implementation at https://github.com/ProfMJSimpson/PenalizedLikelihood
%
%Each state is represented by a B-spline f_k(t) = sum_j c_kj B_kj(t). The
%method alternates between
%   (1) updating the spline coefficients c_k by solving the stacked linear
%       least-squares system [w_d*A; w_m*A'] c = [w_d*b; w_m*b'], balancing
%       data matching against enforcement of the ODE on a fine grid;
%   (2) updating the weights w_d = |1/l_d| and w_m = |1/l_m|;
%   (3) updating beta by maximizing the penalized log-likelihood
%       l(beta) = w_d*l_d + w_m*l_m.
%The ODE is never integrated during estimation: only the splines and the
%right-hand side evaluated at the splines are used.
%
%Two extensions relative to the tutorial are required to fit the EGM
%setting, and both reduce to the tutorial scheme when they are inactive:
%   - Partially observed systems. Hidden states carry a B-spline as well,
%     but contribute no data rows; their coefficients follow from the model
%     rows alone. Since the derivative operator annihilates constants, that
%     block is rank deficient by one per hidden state, so a single weak row
%     anchoring f_k(t_1) to the supplied initial condition is appended
%     (weight ICWeightGP). Fully observed systems never use this row.
%   - State scaling. EGM systems mix states of very different magnitude, so
%     the model residual of state k is divided by its scale (its maximum
%     absolute spline value). Set ScaleModelGP=false for the unscaled
%     residual of the tutorial.
%
%See also EGM, TM, GM, BSPLINE.

    properties (SetAccess = private)
        data                                                                % data aggregate struct
        system                                                              % nested object controlling the ODE system
        settings                                                            % hyperparameters and user input

        T                                                                   % number of data time points
        L                                                                   % number of observed states
        N                                                                   % population size
        K                                                                   % system dimension
        P                                                                   % number of model parameters

        M                                                                   % number of ODE enforcement points
        t_ode                                                               % ODE enforcement grid
        scaled_data                                                         % time grids scaled to [0, 1]
        scaled_ode
        scaled_fine

        splines                                                             % nested BSpline objects (one per state)
        B                                                                   % bases evaluated at the data grid
        dB
        B_ode                                                               % at the ODE enforcement grid
        dB_ode
        B_fine                                                              % at the fine grid for smooth plots
        dB_fine
        delta                                                               % B-spline coefficients per state

        observed                                                            % observed/hidden state indices
        hidden
        column                                                              % state index -> measurement column
        cols                                                                % state index -> coefficient vector columns
        ncoef                                                               % total number of spline coefficients
        nested                                                              % logical for the nested (hidden state) scheme

        x_init                                                              % initial condition guess (hidden state anchor)
        scales                                                              % per-state model residual scales

        beta_gp                                                             % model parameter estimate
        beta_iterations                                                     % parameter estimate per iteration
        varbeta                                                             % uncertainty estimates of beta

        sigma2                                                              % additive measurement error variance
        tau2                                                                % multiplicative measurement error variance
        variances_gp                                                        % measurement error variances along time grid

        w_d                                                                 % data matching weight
        w_m                                                                 % model matching weight
        ell_d                                                               % data matching log-likelihood
        ell_m                                                               % model matching criterion

        smoothed_gp                                                         % profiled states and gradients
        dsmoothed_gp
        smoothed_fine
        dsmoothed_fine
        discrepancy                                                         % xi(t; beta) on the enforcement grid
    end


    methods
        function obj = GP(data, system, settings, beta_init, x_init) % Constructor
            obj.data = data;                                                % experimental data
            obj.system = system;                                            % ODE system

            defaults = struct('order', 4, 'M', 201, 'niter', 20, 'tol', 1e-4, ...
                              'ninner', 8, 'ic_weight', .1, 'scale_model', true, ...
                              'init_mode', 'integrate', 'positive', true);
            fields = fieldnames(defaults);                                  % allow settings structs assembled by hand
            for f = 1:numel(fields)
                if ~isfield(settings, fields{f})
                    settings.(fields{f}) = defaults.(fields{f});
                end
            end
            obj.settings = settings;                                        % hyperparameters

            [obj.T, obj.L, obj.N] = size(data.traces);                      % extract from trajectory dimensions
            obj.K = system.K;
            obj.P = system.P;

            obj.observed = reshape(data.observed, 1, []);                   % index bookkeeping for partial observation
            obj.hidden = find(~ismember(1:obj.K, obj.observed));
            obj.column = zeros(1, obj.K);
            obj.column(obj.observed) = 1:obj.L;

            obj.x_init = reshape(x_init, 1, []);

            obj.M = settings.M;                                             % ODE enforcement grid (tutorial: K = 1001)
            obj.t_ode = linspace(data.t(1), data.t(end), obj.M);

            span = range(data.t);                                           % B-spline bases operate on [0, 1]
            obj.scaled_data = (data.t - data.t(1)) / span;
            obj.scaled_ode = (obj.t_ode - data.t(1)) / span;
            obj.scaled_fine = (data.t_fine - data.t(1)) / span;

            obj.splines = cell(1, obj.K);                                   % one spline per state, including hidden ones
            [obj.B, obj.dB, obj.B_ode, obj.dB_ode, ...
                    obj.B_fine, obj.dB_fine, obj.delta] = deal(cell(1, obj.K));

            for k = 1:obj.K                                                 % collocation matrices (chain rule for derivatives)
                obj.splines{k} = BSpline(settings.order, settings.knots{k});
                obj.B{k} = obj.splines{k}.basis(obj.scaled_data)';
                obj.dB{k} = obj.splines{k}.basis_diff(obj.scaled_data)' / span;
                obj.B_ode{k} = obj.splines{k}.basis(obj.scaled_ode)';
                obj.dB_ode{k} = obj.splines{k}.basis_diff(obj.scaled_ode)' / span;
                obj.B_fine{k} = obj.splines{k}.basis(obj.scaled_fine)';
                obj.dB_fine{k} = obj.splines{k}.basis_diff(obj.scaled_fine)' / span;
                obj.delta{k} = zeros(obj.splines{k}.card, 1);
            end

            obj.cols = cell(1, obj.K);                                      % coefficient vector layout
            offset = 0;
            for k = 1:obj.K
                obj.cols{k} = offset + (1:obj.splines{k}.card);
                offset = offset + obj.splines{k}.card;
            end
            obj.ncoef = offset;
                                                                            % the tutorial scheme assumes every state is
            obj.nested = ~isempty(obj.hidden);                              % observed; with unobserved states the nested
            if isfield(settings, 'nested'), obj.nested = logical(settings.nested); end

            obj.beta_gp = reshape(beta_init, 1, []);
            obj.beta_iterations = obj.beta_gp;
            obj.varbeta = eye(obj.P);
            obj.variances_gp = data.traces.^2 + .1*mean(data.traces).^2;    % initialize mostly multiplicative
            obj.sigma2 = zeros(1, obj.L);
            obj.tau2 = zeros(1, obj.L);
            obj.scales = ones(1, obj.K);
            obj.w_d = 1;
            obj.w_m = 1;
        end


        function output = optimize(obj)                                 % Main optimization function
            obj.initialize();                                               % over-fitted splines (tutorial Figure 3a)
            obj.update_scales();                                            % fixed normalization: recomputing these each
                                                                            % iteration would let a hidden state inflate
                                                                            % itself to make its own equation cheap to
                                                                            % violate, so the scales are set once here

            obj.w_d = 1;                                                    % first pass: w_d = 1, w_m = |1/l_m|
            obj.ell_m = obj.loglik_model(obj.beta_gp);
            obj.w_m = obj.weight(obj.ell_m);

            obj.update_parameters();                                        % beta^(1) from model matching alone
            if obj.nested, obj.inner_solve(obj.beta_gp); end                % settle the states before profiling starts
            obj.beta_iterations = obj.beta_gp;

            for iter = 1:obj.settings.niter
                fprintf('%d ', iter)
                beta_old = obj.beta_gp;

                if obj.nested                                               % step 1: update the spline coefficients
                    obj.inner_solve(obj.beta_gp);                           % jointly over all states (nested form) or
                else                                                        % state by state (tutorial form)
                    obj.update_coefficients();
                end
                obj.estimate_variances();                                   % update variance estimates

                obj.ell_d = obj.loglik_data();                              % step 2: update weights
                obj.ell_m = obj.loglik_model(obj.beta_gp);
                obj.w_d = obj.weight(obj.ell_d);
                obj.w_m = obj.weight(obj.ell_m);

                if obj.nested                                               % step 3: update parameter estimates against
                    obj.update_parameters_data();                           % the data fit of the profiled states, or
                else                                                        % against the penalized likelihood
                    obj.update_parameters();
                end
                obj.beta_iterations(end+1, :) = obj.beta_gp;                %#ok<AGROW>

                if ~mod(iter, 10), fprintf('(%.3e)\n', eucl_rel(beta_old, obj.beta_gp)), end

                if eucl_rel(beta_old, obj.beta_gp) < obj.settings.tol || iter == obj.settings.niter
                    fprintf('(%.3e)\n', eucl_rel(beta_old, obj.beta_gp))
                    break
                end
            end

            obj.save_estimates();                                           % save estimates and profiled trajectories
            output = obj.data;                                              % return data appended with GP results
        end


        function initialize(obj)                                        % Over-fitting splines for the first iteration
            for k = obj.observed                                            % observed states: weighted LS fit of the data
                l = obj.column(k);
                y = reshape(obj.data.traces(:, l, :), obj.T*obj.N, 1);
                Bk = repmat(obj.B{k}, obj.N, 1);
                W = 1 ./ max(reshape(obj.variances_gp(:, l, :), obj.T*obj.N, 1), 1e-10);
                obj.delta{k} = obj.solve_ls(Bk' * (W .* Bk), Bk' * (W .* y), obj.delta{k});
            end

            if isempty(obj.hidden), return, end

            traj = [];                                                      % hidden states carry no data: start either from a
            if strcmpi(obj.settings.init_mode, 'integrate')                 % single ODE solve (initialization only, the
                try                                                         % estimation itself never integrates) ...
                    int_data = obj.data;
                    int_data.init = obj.x_init;
                    traj = obj.system.integrate(obj.beta_gp, int_data, obj.t_ode);
                catch ME
                    disp(ME)
                    traj = [];
                end
            end

            for k = obj.hidden
                if isempty(traj)                                            % ... or from a constant at the supplied IC
                    obj.delta{k} = obj.x_init(k) * ones(obj.splines{k}.card, 1);
                else
                    obj.delta{k} = obj.solve_ls(obj.B_ode{k}, traj(:, k, 1), ...
                                                obj.x_init(k) * ones(obj.splines{k}.card, 1));
                end
            end
        end


        function [X, dX] = states_ode(obj)                              % Splines and gradients on the enforcement grid
            [X, dX] = deal(zeros(obj.M, obj.K));
            for k = 1:obj.K
                X(:, k) = obj.B_ode{k} * obj.delta{k};
                dX(:, k) = obj.dB_ode{k} * obj.delta{k};
            end
            if obj.settings.positive
                X = max(X, 1e-12);                                          % keep the RHS evaluable for positive systems
            end
        end


        function Xd = states_data(obj)                                  % Splines at the measurement time points
            Xd = zeros(obj.T, obj.L);
            for k = obj.observed
                Xd(:, obj.column(k)) = obj.B{k} * obj.delta{k};
            end
            if obj.settings.positive
                Xd = max(Xd, 1e-12);
            end
        end


        function update_scales(obj)                                     % Per-state scales for the model residual
            if ~obj.settings.scale_model
                obj.scales = ones(1, obj.K);                                % unscaled residual as in the tutorial
                return
            end
            X = zeros(obj.M, obj.K);
            for k = 1:obj.K, X(:, k) = obj.B_ode{k} * obj.delta{k}; end
            obj.scales = max(max(abs(X), [], 1), 1e-8);
        end


        function [lm, xi] = loglik_model(obj, beta)                     % Model matching criterion (tutorial Eq. 19)
            [X, dX] = obj.states_ode();
            xi = dX - obj.system.rhs(X, obj.t_ode, reshape(beta, 1, []));   % discrepancy xi(t; beta)
            lm = -sum((xi ./ obj.scales).^2, 'all') / obj.M;
        end


        function ld = loglik_data(obj)                                  % Data matching log-likelihood (tutorial Eq. 11)
            predicted = repmat(obj.states_data(), 1, 1, obj.N);             % with the EGM additive/multiplicative noise model
            variances = max(obj.variances_gp, 1e-10);
            ld = -.5 * sum(log(2*pi*variances) ...
                           + (obj.data.traces - predicted).^2 ./ variances, 'all');
        end


        function update_parameters(obj)                                 % Update model parameter estimates
            [X, dX] = obj.states_ode();                                     % the EGM systems are linear in the parameters,
            design = obj.system.g(X, obj.t_ode);                            % f(x; beta) = g(x)*beta + h(x), so the model
            const = obj.system.h(X, obj.t_ode);                             % matching criterion is quadratic in beta and
            response = flatten(dX - const);                                 % maximizing it is a bounded QP (cf. GM.estimate)

                                                                            % note l_d does not depend on beta (only on the
                                                                            % variances), so maximizing w_d*l_d + w_m*l_m
                                                                            % over beta reduces to maximizing l_m; the
                                                                            % weights act through the coefficient update
            weights = flatten(repmat(1 ./ (sqrt(obj.M) * obj.scales), obj.M, 1));
            design = weights .* design;
            response = weights .* response;

            if isempty(obj.hidden)                                          % fully observed: plain bounded least squares
                obj.beta_gp = obj.bounded_qp(design, response, 1);
                return
            end
                                                                            % otherwise profile the hidden coefficients out
            anchor = obj.hidden_anchor(X);                                  % rather than alternating against them: holding
            design = [design; zeros(numel(obj.hidden), obj.P)];             % them fixed lets an unobserved state that
            response = [response; anchor];                                  % cannot follow the ODE bias beta

            beta = obj.beta_gp;                                             % the projection depends on beta, so a single
            value = obj.profiled_objective(X, design, response, beta);      % projected solve is only locally valid: iterate
                                                                            % Gauss-Newton style with backtracking instead
            for inner = 1:obj.settings.ninner
                basis = orth(obj.hidden_jacobian(X, beta));
                if isempty(basis), break, end
                projected_design = design - basis * (basis' * design);
                projected_response = response - basis * (basis' * response);

                candidate = obj.bounded_qp(projected_design, projected_response, 1);

                step = 1;                                                   % backtrack on the true profiled objective
                accepted = false;
                for trial = 1:5
                    proposal = (1-step) * beta + step * candidate;
                    proposed_value = obj.profiled_objective(X, design, response, proposal);
                    if proposed_value < value
                        accepted = true;
                        break
                    end
                    step = step / 2;
                end
                if ~accepted, break, end

                converged = eucl_rel(beta, proposal) < obj.settings.tol;
                beta = proposal;
                value = proposed_value;
                if converged, break, end
            end

            obj.beta_gp = beta;
        end


        function value = profiled_objective(obj, X, design, response, beta) % Model criterion after profiling out x_hidden
            basis = orth(obj.hidden_jacobian(X, beta));                     % residual directions the hidden splines can
            residual = response - design * reshape(beta, [], 1);            % absorb carry no information about beta
            if ~isempty(basis)
                residual = residual - basis * (basis' * residual);
            end
            value = sum(residual.^2);
        end


        function anchor = hidden_anchor(obj, X)                         % Initial condition rows for the hidden states
            anchor = zeros(numel(obj.hidden), 1);
            for idx = 1:numel(obj.hidden)
                k = obj.hidden(idx);
                anchor(idx) = obj.settings.ic_weight / obj.scales(k) * (X(1, k) - obj.x_init(k));
            end
        end


        function A = hidden_jacobian(obj, X, beta)                      % Residual sensitivity to the hidden coefficients
            J = obj.system.df(X, obj.t_ode, reshape(beta, 1, []));          % d f_s / d x_k along the enforcement grid
            nhidden = numel(obj.hidden);
            cards = cellfun(@numel, obj.delta(obj.hidden));

            A = zeros(obj.K*obj.M + nhidden, sum(cards));
            offset = 0;

            for idx = 1:nhidden
                k = obj.hidden(idx);
                Bk = obj.B_ode{k};
                columns = offset + (1:cards(idx));

                for s = 1:obj.K                                             % d(residual_s)/d(c_k), matching update_hidden
                    rows = (s-1)*obj.M + (1:obj.M);
                    block = -(J(rows, k) .* Bk);
                    if s == k, block = block + obj.dB_ode{k}; end
                    A(rows, columns) = block / (sqrt(obj.M) * obj.scales(s));
                end
                                                                            % the initial condition anchor constrains the
                A(obj.K*obj.M + idx, columns) = ...                         % hidden coefficients too, so it joins the
                    obj.settings.ic_weight / obj.scales(k) * Bk(1, :);      % system being profiled out
                offset = offset + cards(idx);
            end
        end


        function c = coefficients(obj)                                  % Pack the spline coefficients into one vector
            c = zeros(obj.ncoef, 1);
            for k = 1:obj.K, c(obj.cols{k}) = obj.delta{k}; end
        end


        function set_coefficients(obj, c)                               % Unpack them again
            for k = 1:obj.K, obj.delta{k} = c(obj.cols{k}); end
        end


        function [r, Jc, Jb] = inner_residual(obj, beta)                % Inner criterion of Hooker et al. (2011), Eq. 3.2
            beta = reshape(beta, 1, []);                                    % stacked data and model matching residuals,
            [X, dX] = obj.states_ode();                                     % with derivatives wrt every spline coefficient
            Xd = obj.states_data();                                         % (all states at once, not state by state) and
            F = obj.system.rhs(X, obj.t_ode, beta);                         % wrt the model parameters
            ndata = obj.T * obj.N * obj.L;
            nmodel = obj.K * obj.M;
            r = zeros(ndata + nmodel + numel(obj.hidden), 1);

            if nargout > 1
                J = obj.system.df(X, obj.t_ode, beta);
                G = obj.system.g(X, obj.t_ode);
                Jc = zeros(numel(r), obj.ncoef);
                Jb = zeros(numel(r), obj.P);
            end

            for l = 1:obj.L                                                 % data matching rows
                k = obj.observed(l);
                rows = (l-1)*obj.T*obj.N + (1:obj.T*obj.N);
                y = reshape(obj.data.traces(:, l, :), [], 1);
                variances = reshape(obj.variances_gp(:, l, :), [], 1);
                wd = obj.w_d ./ sqrt(max(variances, 1e-10));
                r(rows) = wd .* (repmat(Xd(:, l), obj.N, 1) - y);
                if nargout > 1
                    Jc(rows, obj.cols{k}) = wd .* repmat(obj.B{k}, obj.N, 1);
                end
            end

            for st = 1:obj.K                                                % model matching rows
                rows = ndata + (st-1)*obj.M + (1:obj.M);
                ws = obj.w_m / (sqrt(obj.M) * obj.scales(st));
                r(rows) = ws * (dX(:, st) - F(:, st));
                if nargout > 1
                    for k = 1:obj.K
                        block = -(J((st-1)*obj.M + (1:obj.M), k) .* obj.B_ode{k});
                        if st == k, block = block + obj.dB_ode{k}; end
                        Jc(rows, obj.cols{k}) = ws * block;
                    end
                    Jb(rows, :) = -ws * G((st-1)*obj.M + (1:obj.M), :);
                end
            end

            for idx = 1:numel(obj.hidden)                                   % weak initial condition anchors
                k = obj.hidden(idx);
                row = ndata + nmodel + idx;
                wic = obj.settings.ic_weight * obj.w_m / obj.scales(k);
                r(row) = wic * (X(1, k) - obj.x_init(k));
                if nargout > 1, Jc(row, obj.cols{k}) = wic * obj.B_ode{k}(1, :); end
            end
        end


        function inner_solve(obj, beta)                                 % Inner problem: C(beta) by Gauss-Newton
            for iter = 1:obj.settings.ninner
                [r, Jc] = obj.inner_residual(beta);
                value = sum(r.^2);
                direction = -obj.normal_solve(Jc, r);

                current = obj.coefficients();                               % backtrack so a bad linearization cannot
                step = 1;                                                   % throw the splines away
                accepted = false;
                for trial = 1:5
                    obj.set_coefficients(current + step * direction);
                    if sum(obj.inner_residual(beta).^2) < value
                        accepted = true;
                        break
                    end
                    step = step / 2;
                end
                if ~accepted
                    obj.set_coefficients(current);
                    break
                end
                if eucl_rel(current, obj.coefficients()) < obj.settings.tol, break, end
            end
        end


        function x = normal_solve(obj, J, r)                            % Regularized Gauss-Newton normal equations
            A = J' * J;
            A = (A + A') / 2;
            ridge = 1e-9 * max(1e-30, trace(A) / size(A, 1));               % keep the system conditioned when a hidden
            x = obj.solve_ls(A + ridge * eye(size(A)), J' * r, zeros(size(A, 1), 1));
        end


        function mixed = mixed_partial(obj, beta)                       % d2J/(dc dtheta) at the current coefficients
            beta = reshape(beta, 1, []);                                    % dJ/dc = 2*Jc'*r is analytic, so difference
            mixed = zeros(obj.ncoef, obj.P);                                % it in theta at fixed c: 2P residual builds,
            for q = 1:obj.P                                                 % no inner solves, and no noise from
                step = 1e-6 * max(1e-6, abs(beta(q)));                      % differencing through the solver
                plus = beta; plus(q) = plus(q) + step;
                minus = beta; minus(q) = minus(q) - step;
                [rp, Jp] = obj.inner_residual(plus);
                [rm, Jm] = obj.inner_residual(minus);
                mixed(:, q) = (Jp' * rp - Jm' * rm) / (2 * step);
            end
        end


        function update_parameters_data(obj)                            % Outer criterion of Hooker et al. (2011), Eq. 3.4
            ndata = obj.T * obj.N * obj.L;                                  % beta is chosen so that the PROFILED states
            [r, Jc] = obj.inner_residual(obj.beta_gp);                      % fit the data. The model residual does not
                                                                            % enter here, which is what makes unobserved
            A = Jc' * Jc;                                                   % states harmless: they are nuisance
            A = (A + A') / 2;                                               % quantities of the inner problem only
            ridge = 1e-9 * max(1e-30, trace(A) / size(A, 1));
            mixed = obj.mixed_partial(obj.beta_gp);                         % implicit function theorem. d2J/dc2 by
            dcdbeta = -(A + ridge * eye(size(A))) \ mixed;                  % Gauss-Newton is accurate enough, but
                                                                            % d2J/(dc dtheta) is NOT: Gauss-Newton drops
                                                                            % sum_i r_i d2r_i/(dc dtheta), a leading term
                                                                            % here since the ODE residual is not small

            design = Jc(1:ndata, :) * dcdbeta;                              % d(data residual)/d(beta)
            residual = r(1:ndata);
            value = sum(residual.^2);

            beta = reshape(obj.beta_gp, [], 1);                             % Gauss-Newton step, expressed in absolute
            candidate = obj.bounded_qp(design, design * beta - residual, 1);% beta so the bounds and prior still apply

            current = obj.coefficients();                                   % backtrack on the true outer criterion, which
            accepted = false;                                               % means re-solving the inner problem
            step = 1;
            for trial = 1:4
                proposal = (1-step) * obj.beta_gp + step * candidate;
                obj.set_coefficients(current);
                obj.inner_solve(proposal);
                trial_residual = obj.inner_residual(proposal);
                if sum(trial_residual(1:ndata).^2) < value
                    obj.beta_gp = proposal;
                    accepted = true;
                    break
                end
                step = step / 2;
            end
            if ~accepted, obj.set_coefficients(current); end
        end


        function opt = bounded_qp(obj, X, Y, weights)                   % Weighted LS with box constraints and prior
            prior = obj.settings.prior;                                     % (diagonally weighted variant of
            lb = reshape(obj.settings.lb, 1, []);                           %  Optimization.QPGLS, avoiding the dense
            ub = reshape(obj.settings.ub, 1, []);                           %  inverse of a (K*M)-by-(K*M) covariance)

            H = X' * (weights .* X) + prior.prec;
            H = (H + H') / 2;                                               % force symmetry
            f = -X' * (weights .* Y) - prior.prec * prior.mean;

            A = kron([-1; 1], eye(length(lb)));                             % bounds to inequality constraints
            b = [-lb ub]';
            Aeq = zeros(0, length(lb));                                     % empty equality constraints
            beq = zeros(0, 1);
            try
                opt = mpcActiveSetSolver(H, f, A, b, Aeq, beq, false(2*length(lb), 1), ...
                                         Optimization.options_mpcActiveSet)';
            catch ME
                disp(ME)
                H = nearestSPD(H);                                          % retry if H is not SPD
                opt = mpcActiveSetSolver(H, f, A, b, Aeq, beq, false(2*length(lb), 1), ...
                                         Optimization.options_mpcActiveSet)';
            end

            if any(~isfinite(opt)), opt = obj.beta_gp; end                  % guard against solver failure
            opt = min(max(opt, lb), ub);
        end


        function update_coefficients(obj)                               % Update spline coefficients
            X = obj.states_ode();                                           % right-hand side at the current splines
            rhs = obj.system.rhs(X, obj.t_ode, obj.beta_gp);

            for k = obj.observed                                            % observed states: tutorial update (Eq. 10)
                obj.update_observed(k, rhs);
            end

            for k = obj.hidden                                              % hidden states: coupled update, refreshing the
                obj.update_hidden(k);                                       % splines in between (Gauss-Seidel sweep)
            end
        end


        function update_observed(obj, k, rhs)                           % Stacked data/model system (tutorial Eq. 10)
            wm = obj.w_m / sqrt(obj.M) / obj.scales(k);                     % model rows normalized by sqrt(M) so that the
            design = wm * obj.dB_ode{k};                                    % data/model balance is grid size independent
            response = wm * rhs(:, k);

            l = obj.column(k);                                              % data rows, weighted by the estimated
            y = reshape(obj.data.traces(:, l, :), obj.T*obj.N, 1);          % measurement error standard deviations
            variances = reshape(obj.variances_gp(:, l, :), obj.T*obj.N, 1);
            wd = obj.w_d ./ sqrt(max(variances, 1e-10));

            design = [design; wd .* repmat(obj.B{k}, obj.N, 1)];
            response = [response; wd .* y];

            obj.delta{k} = obj.solve_ls(design, response, obj.delta{k});
        end


        function update_hidden(obj, k)                                  % Coupled update for an unobserved state
            [X, dX] = obj.states_ode();                                     % a hidden state carries no data rows, so the
            F = obj.system.rhs(X, obj.t_ode, obj.beta_gp);                  % only information reaching it comes through
            J = obj.system.df(X, obj.t_ode, obj.beta_gp);                   % the equations it appears in. Freezing the
            Bk = obj.B_ode{k};                                              % other states and linearizing f in x_k gives
            xk = X(:, k);                                                   % a Gauss-Newton step using every such equation
                                                                            % (the tutorial update keeps equation k only,
            [design, response] = deal([]);                                  %  which leaves x_k undetermined by the data)
            for s = 1:obj.K
                dfs_dxk = J((s-1)*obj.M + (1:obj.M), k);                    % d f_s / d x_k along the enforcement grid
                if s ~= k && max(abs(dfs_dxk)) <= 1e-12, continue, end      % skip equations not involving x_k

                ws = obj.w_m / sqrt(obj.M) / obj.scales(s);
                if s == k                                                   % own equation: dB_k c - f_k(x_k)
                    design = [design; ws * (obj.dB_ode{k} - dfs_dxk .* Bk)];        %#ok<AGROW>
                    response = [response; ws * (F(:, k) - dfs_dxk .* xk)];          %#ok<AGROW>
                else                                                        % other equations: dX_s - f_s(x_k)
                    design = [design; ws * (dfs_dxk .* Bk)];                        %#ok<AGROW>
                    response = [response; ws * (dX(:, s) - F(:, s) + dfs_dxk .* xk)];%#ok<AGROW>
                end
            end

            wic = obj.settings.ic_weight * obj.w_m / obj.scales(k);         % weak initial condition anchor, regularizing
            design = [design; wic * Bk(1, :)];                              % the constant direction left free when x_k
            response = [response; wic * obj.x_init(k)];                     % enters the system only through derivatives

            proposed = obj.solve_ls(design, response, obj.delta{k});
            obj.delta{k} = obj.damped_step(k, proposed);
        end


        function coefficients = damped_step(obj, k, proposed)           % Backtrack a Gauss-Newton step if it overshoots
            current = obj.delta{k};
            best = current;
            reference = obj.model_residual();
            step = 1;
            for trial = 1:5                                                 % halve until the model residual decreases
                obj.delta{k} = (1-step) * current + step * proposed;
                if obj.model_residual() < reference
                    best = obj.delta{k};
                    break
                end
                step = step / 2;
            end
            obj.delta{k} = best;
            coefficients = best;
        end


        function ss = model_residual(obj)                               % Scaled model discrepancy at current splines
            [X, dX] = obj.states_ode();
            xi = dX - obj.system.rhs(X, obj.t_ode, obj.beta_gp);
            ss = sum(((xi ./ obj.scales).^2), 'all');
        end


        function coefficients = solve_ls(~, design, response, previous)  % Least squares with a pseudoinverse fallback
            try
                coefficients = design \ response;
            catch ME
                disp(ME)
                coefficients = pinv(full(design)) * response;
            end
            if any(~isfinite(coefficients)), coefficients = previous; end
        end


        function estimate_variances(obj)                                % Update measurement error variances
            Xd = obj.states_data();
            for k = 1:obj.L
                predicted = repmat(Xd(:, k), 1, 1, obj.N);                  % profiled splines at the data time points
                design = [ones(obj.N*obj.T, 1) flatten(predicted).^2];      % columns for additive and multiplicative noise
                response = flatten(predicted - obj.data.traces(:, k, :)).^2;

                coefficients = lsqnonneg(design, response)';                % initialize nonzero LS estimates for noise parameters
                if sum(coefficients) == 0, coefficients(1) = mean(response); end
                                                                            % optimize further iteratively
                coefficients = Optimization.noise_parameters(coefficients, predicted, obj.data.traces(:, k, :));
                if sum(coefficients) == 0, coefficients(1) = mean(response); end

                obj.sigma2(k) = coefficients(1);                            % store optimum and compute variances
                obj.tau2(k) = coefficients(2);

                obj.variances_gp(:, k, :) = reshape(design * coefficients', obj.T, 1, obj.N);
            end
        end


        function w = weight(~, ell)                                     % Iterative weight w = |1/l| (tutorial Section 3.1)
            w = abs(1 / ell);
            if ~isfinite(w) || w == 0, w = 1; end
        end


        function save_estimates(obj)                                    % Extract results
            [obj.smoothed_gp, obj.dsmoothed_gp] = deal(zeros(obj.T, obj.K));
            [obj.smoothed_fine, obj.dsmoothed_fine] = deal(zeros(obj.data.T_fine, obj.K));

            init = zeros(1, obj.K);                                         % initial conditions as a post-processing step:
            for k = 1:obj.K                                                 % simply evaluate each spline at t_1
                obj.smoothed_gp(:, k) = obj.B{k} * obj.delta{k};
                obj.dsmoothed_gp(:, k) = obj.dB{k} * obj.delta{k};
                obj.smoothed_fine(:, k) = obj.B_fine{k} * obj.delta{k};
                obj.dsmoothed_fine(:, k) = obj.dB_fine{k} * obj.delta{k};
                init(k) = obj.B{k}(1, :) * obj.delta{k};
            end

            if obj.settings.positive
                obj.smoothed_gp = max(obj.smoothed_gp, 1e-12);              % clip nonpositive values
                obj.smoothed_fine = max(obj.smoothed_fine, 1e-12);
                init = max(init, 1e-12);
            end

            [~, obj.discrepancy] = obj.loglik_model(obj.beta_gp);

            obj.data.init = init;                                           % store results
            obj.data.beta_gp = obj.beta_gp;
            obj.data.beta_iterations = obj.beta_iterations;
            obj.data.varbeta = obj.varbeta;
            obj.data.smoothed_gp = obj.smoothed_gp;
            obj.data.dsmoothed_gp = obj.dsmoothed_gp;
            obj.data.smoothed_fine = obj.smoothed_fine;
            obj.data.dsmoothed_fine = obj.dsmoothed_fine;
            obj.data.t_ode = obj.t_ode;
            obj.data.discrepancy = obj.discrepancy;
            obj.data.weights_gp = [obj.w_d obj.w_m];
            obj.data.loglik_gp = [obj.ell_d obj.ell_m];
            obj.data.variances_gp = obj.variances_gp;
            obj.data.delta_gp = obj.delta;
        end
    end
end
