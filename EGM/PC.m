classdef PC < handle                                                    % Parameter cascade (generalized profiling)
%PC implements generalized profiling as originally formulated by Ramsay,
%Hooker, Campbell and Cao (2007), JRSS-B 69(5), 741-796, with the penalty
%weight structure used by Tyuryaev et al. (github.com/vadimtyuryaev/
%Beyond-NLS-Inference).
%
%This is the original method, not the tutorial simplification implemented
%in GP. Three things differ, and all three matter:
%
%  1. The inner criterion is minimized over ALL spline coefficients at once
%     (Ramsay Eq. 15). The right-hand side f is evaluated at the current
%     splines as a function of c, never frozen, so the problem is genuinely
%     nonlinear in c and is solved by Gauss-Newton. GP's tutorial path
%     freezes f and solves one linear system per state, which drops the
%     cross-state blocks of the Jacobian.
%
%  2. The outer criterion H(theta) is a measure of fit to the DATA ONLY
%     (Ramsay Section 2.1): the ODE penalty does not appear in it. Because
%     c_hat(theta) is already regularized by the penalty, H needs no
%     further regularization. This is what makes unobserved states
%     harmless: they are nuisance parameters of the inner problem.
%
%  3. dH/dtheta is obtained analytically by the implicit function theorem
%     (Ramsay Eqs. 16-17),
%         dc_hat/dtheta = -[d2J/dc2]^-1 [d2J/dc dtheta],
%         dH/dtheta     = dH/dtheta|_explicit + (dH/dc)(dc_hat/dtheta),
%     using the Gauss-Newton approximations d2J/dc2 ~ 2*Jc'Jc and
%     d2J/(dc dtheta) ~ 2*Jc'Jb. Here dH/dtheta|_explicit = 0, since theta
%     reaches H only through c_hat.
%
%The penalty follows Ramsay Eqs. 11-13, PEN_s = integral of the squared
%discrepancy, with per-equation multipliers lambda_s. The integral is
%evaluated by the trapezoidal rule rather than a bare sum, so lambda does
%not have to be retuned when the enforcement grid is refined. Unobserved
%states enter the penalty (the sum runs over all K equations) but not the
%data term (which runs over the observed subset only), exactly as in
%Ramsay Section 1.1 for the "distributed partial data problem".
%
%Two residual blocks are taken from Tyuryaev et al. rather than Ramsay: a
%weak initial condition penalty, and a positivity barrier min(x, 0) on the
%splines. The barrier is non-smooth at zero, but the EGM systems leave the
%positive orthant invariant, so it is inactive at the solution and the
%Gauss-Newton Hessian is well defined there.
%
%See also EGM, GP, TM, MAGI.

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
        quadrature                                                          % trapezoidal weights on that grid

        splines                                                             % nested BSpline objects (one per state)
        B                                                                   % bases at the data grid
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

        x_init                                                              % initial condition guess
        scales                                                              % per-state magnitudes used to scale lambda
        lambda                                                              % realized penalty weights

        beta_pc                                                             % model parameter estimate
        beta_iterations                                                     % parameter estimate per iteration
        varbeta                                                             % Gauss-Newton covariance approximation

        sigma2                                                              % additive measurement error variance
        tau2                                                                % multiplicative measurement error variance
        variances_pc                                                        % measurement error variances along time grid

        H_value                                                             % outer criterion at the current estimate
        J_value                                                             % inner criterion at the current estimate

        smoothed_pc                                                         % profiled states and gradients
        dsmoothed_pc
        smoothed_fine
        dsmoothed_fine
        discrepancy                                                         % xi(t; beta) on the enforcement grid
    end


    methods
        function obj = PC(data, system, settings, beta_init, x_init)   % Constructor
            obj.data = data;                                                % experimental data
            obj.system = system;                                            % ODE system

            defaults = struct('order', 4, 'M', 201, 'niter', 20, 'tol', 1e-4, ...
                              'ninner', 20, 'inner_tol', 1e-8, 'positive', true, 'fd_step', 1e-6, ...
                              'nsearch', 8, 'nstart', 5, 'solver', 'fmincon', ...
                              'lambda', 1e2, 'lambda_data', 1, 'lambda_ic', 1e-2, ...
                              'lambda_positivity', 1e3, 'anchor_all', false, ...
                              'init_mode', 'integrate');
            fields = fieldnames(defaults);                                  % allow settings assembled by hand
            for f = 1:numel(fields)
                if ~isfield(settings, fields{f}), settings.(fields{f}) = defaults.(fields{f}); end
            end
            obj.settings = settings;

            [obj.T, obj.L, obj.N] = size(data.traces);
            obj.K = system.K;
            obj.P = system.P;

            obj.observed = reshape(data.observed, 1, []);
            obj.hidden = find(~ismember(1:obj.K, obj.observed));
            obj.column = zeros(1, obj.K);
            obj.column(obj.observed) = 1:obj.L;
            obj.x_init = reshape(x_init, 1, []);

            obj.M = settings.M;
            obj.t_ode = linspace(data.t(1), data.t(end), obj.M);
            obj.quadrature = PC.trapezoid(obj.t_ode);                       % PEN is an integral, not a sum

            span = range(data.t);                                           % B-spline bases operate on [0, 1]
            scaled_data = (data.t - data.t(1)) / span;
            scaled_ode = (obj.t_ode - data.t(1)) / span;
            scaled_fine = (data.t_fine - data.t(1)) / span;

            obj.splines = cell(1, obj.K);
            [obj.B, obj.dB, obj.B_ode, obj.dB_ode, ...
                    obj.B_fine, obj.dB_fine, obj.delta, obj.cols] = deal(cell(1, obj.K));

            offset = 0;
            for k = 1:obj.K
                obj.splines{k} = BSpline(settings.order, settings.knots{k});
                obj.B{k} = obj.splines{k}.basis(scaled_data)';
                obj.dB{k} = obj.splines{k}.basis_diff(scaled_data)' / span;
                obj.B_ode{k} = obj.splines{k}.basis(scaled_ode)';
                obj.dB_ode{k} = obj.splines{k}.basis_diff(scaled_ode)' / span;
                obj.B_fine{k} = obj.splines{k}.basis(scaled_fine)';
                obj.dB_fine{k} = obj.splines{k}.basis_diff(scaled_fine)' / span;
                obj.delta{k} = zeros(obj.splines{k}.card, 1);
                obj.cols{k} = offset + (1:obj.splines{k}.card);
                offset = offset + obj.splines{k}.card;
            end
            obj.ncoef = offset;

            obj.beta_pc = reshape(beta_init, 1, []);
            obj.beta_iterations = obj.beta_pc;
            obj.varbeta = eye(obj.P);
            obj.variances_pc = data.traces.^2 + .1*mean(data.traces).^2;
            obj.sigma2 = zeros(1, obj.L);
            obj.tau2 = zeros(1, obj.L);
            obj.scales = ones(1, obj.K);
        end


        function output = optimize(obj)                                 % Main routine: the parameter cascade
            obj.initialize();                                               % starting splines
            obj.set_lambda();                                               % penalty weights, fixed thereafter

            obj.inner_solve(obj.beta_pc);                                   % c_hat(theta^(0))
            obj.beta_iterations = obj.beta_pc;

            for iter = 1:obj.settings.niter                                 % the outer problem is solved completely by
                fprintf('%d ', iter)                                        % solve_outer, so this loop only re-estimates
                beta_old = obj.beta_pc;                                     % the measurement error variances, which the
                                                                            % data block of J and the criterion H both use
                obj.estimate_variances();
                obj.solve_outer(iter == 1);                                 % multistart on the first pass only
                obj.beta_iterations(end+1, :) = obj.beta_pc;                %#ok<AGROW>

                fprintf('(%.3e)\n', eucl_rel(beta_old, obj.beta_pc))

                if eucl_rel(beta_old, obj.beta_pc) < obj.settings.tol, break, end
            end

            obj.save_estimates();
            output = obj.data;
        end


        function set_lambda(obj)                                        % Per-equation penalty weights (Ramsay Eq. 13)
            X = obj.states(obj.coefficients());                             % the discrepancy of equation s carries units
            obj.scales = max(max(abs(X), [], 1), 1e-8);                     % of x_s per unit time, and PEN_s integrates
            span = range(obj.data.t);                                       % its square over the interval, so x_s^2/span
                                                                            % is its natural size. Dividing by that makes
            base = obj.settings.lambda;                                     % a single lambda comparable across equations
            if isscalar(base), base = repmat(base, 1, obj.K); end           % and independent of the units of each state
            obj.lambda.equation = reshape(base, 1, []) * span ./ obj.scales.^2;

            obj.lambda.data = obj.settings.lambda_data;
            obj.lambda.ic = obj.settings.lambda_ic * span ./ obj.scales.^2;
            obj.lambda.positivity = obj.settings.lambda_positivity ./ obj.scales.^2;

            obj.lambda.anchored = obj.hidden;                               % observed states are pinned by their data
            if obj.settings.anchor_all, obj.lambda.anchored = 1:obj.K; end
        end


        function initialize(obj)                                        % Starting splines
            for k = obj.observed                                            % observed: weighted least squares on the data
                l = obj.column(k);
                y = reshape(obj.data.traces(:, l, :), obj.T*obj.N, 1);
                Bk = repmat(obj.B{k}, obj.N, 1);
                W = 1 ./ max(reshape(obj.variances_pc(:, l, :), obj.T*obj.N, 1), 1e-10);
                obj.delta{k} = PC.solve(Bk' * (W .* Bk), Bk' * (W .* y), obj.delta{k});
            end

            if isempty(obj.hidden), return, end

            traj = [];
            if strcmpi(obj.settings.init_mode, 'integrate')                 % one ODE solve for a starting point only;
                try                                                         % the cascade itself never integrates
                    int_data = obj.data;
                    int_data.init = obj.x_init;
                    traj = obj.system.integrate(obj.beta_pc, int_data, obj.t_ode);
                catch ME
                    disp(ME)
                    traj = [];
                end
            end

            for k = obj.hidden
                if isempty(traj)
                    obj.delta{k} = obj.x_init(k) * ones(obj.splines{k}.card, 1);
                else
                    obj.delta{k} = PC.solve(obj.B_ode{k}' * obj.B_ode{k}, ...
                                            obj.B_ode{k}' * traj(:, k, 1), ...
                                            obj.x_init(k) * ones(obj.splines{k}.card, 1));
                end
            end
        end


        function c = coefficients(obj)                                  % Pack the spline coefficients
            c = zeros(obj.ncoef, 1);
            for k = 1:obj.K, c(obj.cols{k}) = obj.delta{k}; end
        end


        function set_coefficients(obj, c)                               % Unpack them again
            for k = 1:obj.K, obj.delta{k} = c(obj.cols{k}); end
        end


        function [X, dX] = states(obj, c)                               % Splines on the enforcement grid (unclipped)
            [X, dX] = deal(zeros(obj.M, obj.K));
            for k = 1:obj.K
                X(:, k) = obj.B_ode{k} * c(obj.cols{k});
                dX(:, k) = obj.dB_ode{k} * c(obj.cols{k});
            end
        end


        function [r, Jc, Jb] = inner_residual(obj, c, beta)             % Inner criterion J(c | theta, lambda), Eq. 15
            beta = reshape(beta, 1, []);
            [X, dX] = obj.states(c);

            mask = true(size(X));                                           % evaluate f on the positive orthant; the
            Xe = X;                                                         % clip is inactive once the barrier bites,
            if obj.settings.positive                                        % and its derivative is zero where it is not
                mask = X > 1e-12;
                Xe = max(X, 1e-12);
            end

            F = obj.system.rhs(Xe, obj.t_ode, beta);
            ndata = obj.T * obj.N * obj.L;
            nmodel = obj.K * obj.M;
            nanchor = numel(obj.lambda.anchored);
            npos = obj.settings.positive * obj.K * obj.M;
            r = zeros(ndata + nmodel + nanchor + npos, 1);

            if nargout > 1
                Jacobian = obj.system.df(Xe, obj.t_ode, beta);
                G = obj.system.g(Xe, obj.t_ode);
                Jc = zeros(numel(r), obj.ncoef);
                Jb = zeros(numel(r), obj.P);
            end

            for l = 1:obj.L                                                 % data block, over the observed subset only
                k = obj.observed(l);
                rows = (l-1)*obj.T*obj.N + (1:obj.T*obj.N);
                y = reshape(obj.data.traces(:, l, :), [], 1);
                variances = reshape(obj.variances_pc(:, l, :), [], 1);
                w = sqrt(obj.lambda.data ./ max(variances, 1e-10));
                predicted = repmat(obj.B{k} * c(obj.cols{k}), obj.N, 1);
                r(rows) = w .* (predicted - y);
                if nargout > 1, Jc(rows, obj.cols{k}) = w .* repmat(obj.B{k}, obj.N, 1); end
            end

            for s = 1:obj.K                                                 % penalty block, over ALL equations
                rows = ndata + (s-1)*obj.M + (1:obj.M);
                w = sqrt(obj.lambda.equation(s) * obj.quadrature(:));       % trapezoidal quadrature of PEN_s
                r(rows) = w .* (dX(:, s) - F(:, s));
                if nargout > 1
                    for k = 1:obj.K                                         % f depends on every state, so every block
                        block = -(Jacobian((s-1)*obj.M + (1:obj.M), k) .* mask(:, k) .* obj.B_ode{k});
                        if s == k, block = block + obj.dB_ode{k}; end
                        Jc(rows, obj.cols{k}) = w .* block;
                    end
                    Jb(rows, :) = -w .* G((s-1)*obj.M + (1:obj.M), :);
                end
            end

            for idx = 1:nanchor                                             % initial condition block
                k = obj.lambda.anchored(idx);
                row = ndata + nmodel + idx;
                w = sqrt(obj.lambda.ic(k));
                r(row) = w * (obj.B_ode{k}(1, :) * c(obj.cols{k}) - obj.x_init(k));
                if nargout > 1, Jc(row, obj.cols{k}) = w * obj.B_ode{k}(1, :); end
            end

            if obj.settings.positive                                        % positivity barrier on the splines
                for s = 1:obj.K
                    rows = ndata + nmodel + nanchor + (s-1)*obj.M + (1:obj.M);
                    w = sqrt(obj.lambda.positivity(s));
                    negative = X(:, s) < 0;
                    r(rows) = w * min(X(:, s), 0);
                    if nargout > 1, Jc(rows, obj.cols{s}) = w * (negative .* obj.B_ode{s}); end
                end
            end
        end


        function inner_solve(obj, beta)                                 % Minimize J over ALL coefficients jointly
            c = obj.coefficients();
            if PC.available('lsqnonlin') && ~strcmpi(obj.settings.solver, 'builtin')
                options = optimoptions('lsqnonlin', 'Display', 'none', ...   % same globalized Gauss-Newton that
                                       'Algorithm', 'trust-region-reflective', ...  % Tyuryaev et al. use for the inner
                                       'SpecifyObjectiveGradient', true, ...  % problem. The IFT needs dJ/dc = 0 to hold
                                       'MaxIterations', 2000, ...            % accurately, so the tolerances are tight:
                                       'MaxFunctionEvaluations', 1e5, ...    % a loosely converged c_hat silently
                                       'FunctionTolerance', 1e-14, ...       % corrupts the outer gradient
                                       'StepTolerance', 1e-14, ...
                                       'OptimalityTolerance', 1e-14);
                c = lsqnonlin(@(cvec) obj.inner_objective(cvec, beta), c, [], [], options);
            else
                c = obj.damped_gauss_newton(c, beta);                        % fallback: no Optimization Toolbox
            end
            obj.set_coefficients(c);
            obj.J_value = sum(obj.inner_residual(c, beta).^2);
        end


        function [r, Jc] = inner_objective(obj, c, beta)                % lsqnonlin interface to the inner residual
            if nargout > 1
                [r, Jc] = obj.inner_residual(c, beta);
            else
                r = obj.inner_residual(c, beta);
            end
        end


        function c = damped_gauss_newton(obj, c, beta)                  % Fallback inner solver
            for iter = 1:obj.settings.ninner
                [r, Jc] = obj.inner_residual(c, beta);
                value = sum(r.^2);
                direction = -PC.normal_solve(Jc, r);

                step = 1;
                accepted = false;
                for trial = 1:6
                    candidate = c + step * direction;
                    if sum(obj.inner_residual(candidate, beta).^2) < value
                        accepted = true;
                        break
                    end
                    step = step / 2;
                end
                if ~accepted, break, end
                if eucl_rel(c, candidate) < obj.settings.inner_tol, c = candidate; break, end
                c = candidate;
            end
        end


        function [H, K_profiled, residual] = outer(obj, beta)           % Outer criterion H and its profiled Jacobian
            obj.inner_solve(beta);                                          % c_hat(theta)
            c = obj.coefficients();
            [r, Jc, Jb] = obj.inner_residual(c, beta);

            ndata = obj.T * obj.N * obj.L;
            residual = r(1:ndata);                                          % H depends on the data block alone
            H = sum(residual.^2);

            if nargout > 1                                                  % implicit function theorem, Eqs. 16-17:
                mixed = obj.mixed_partial(c, beta);                         % dc_hat/dtheta = -[d2J/dc2]^-1 [d2J/dc dtheta]
                A = Jc' * Jc;                                               % d2J/dc2 by Gauss-Newton is accurate enough,
                A = (A + A') / 2;                                           % but d2J/(dc dtheta) is NOT: see mixed_partial
                ridge = 1e-10 * max(1e-30, trace(A) / size(A, 1));
                dcdbeta = -PC.solve(A + ridge*eye(size(A)), mixed, zeros(obj.ncoef, obj.P));
                K_profiled = Jc(1:ndata, :) * dcdbeta;                      % dH/dtheta = 2*residual'*K_profiled
            end
        end


        function mixed = mixed_partial(obj, c, beta)                    % d2J/(dc dtheta) at fixed c
            beta = reshape(beta, 1, []);                                    % The Gauss-Newton form Jc'*Jb drops the term
            mixed = zeros(obj.ncoef, obj.P);                                % sum_i r_i d2r_i/(dc dtheta), which vanishes
                                                                            % only when the residuals are small. In
            for q = 1:obj.P                                                 % generalized profiling the ODE residual is
                step = obj.settings.fd_step * max(1e-6, abs(beta(q)));      % deliberately NOT small, so that term is a
                plus = beta; plus(q) = plus(q) + step;                      % leading contribution: dropping it gave a
                minus = beta; minus(q) = minus(q) - step;                   % 40% error in dH/dtheta on a 3-state chain.
                                                                            %
                [rp, Jp] = obj.inner_residual(c, plus);                     % dJ/dc = 2*Jc'*r is analytic, so differencing
                [rm, Jm] = obj.inner_residual(c, minus);                    % it in theta at FIXED c is cheap (2P residual
                mixed(:, q) = (Jp' * rp - Jm' * rm) / (2 * step);           % builds, no inner solves) and accurate: it
            end                                                             % never differences through the solver, which
        end                                                                 % is what makes a naive FD outer loop noisy


        function [H, gradient] = outer_with_gradient(obj, beta)         % fmincon interface: H and the analytic dH/dtheta
            if nargout > 1                                                  % one inner solve plus 2P residual builds,
                [H, K_profiled, residual] = obj.outer(beta);                % against P+1 inner solves for a finite
                gradient = reshape(2 * (residual' * K_profiled), size(beta));% difference gradient, and with no noise from
            else                                                            % differencing through the inner solver
                H = obj.outer(beta);
            end
        end


        function solve_outer(obj, multistart)                           % Solve the outer problem for theta
            if nargin < 2, multistart = false; end
            lb = reshape(obj.settings.lb, 1, []);
            ub = reshape(obj.settings.ub, 1, []);

            starts = obj.beta_pc;                                           % H has local minima, and a single bad start
            if multistart && obj.settings.nstart > 1                        % parks parameters on their bounds, so start
                starts = [starts; obj.random_starts(obj.settings.nstart - 1)];  % from several points and keep the best
            end

            current = obj.coefficients();
            best_beta = obj.beta_pc;
            best_H = inf;

            for run = 1:size(starts, 1)
                obj.set_coefficients(current);
                [beta, H] = obj.run_outer(starts(run, :), lb, ub);
                if isfinite(H) && H < best_H
                    best_H = H;
                    best_beta = beta;
                end
            end

            obj.beta_pc = min(max(best_beta, lb), ub);
            obj.H_value = best_H;
            obj.set_coefficients(current);
            obj.inner_solve(obj.beta_pc);                                   % leave the splines at c_hat(theta_hat)

            [~, K_profiled, residual] = obj.outer(obj.beta_pc);
            obj.varbeta = PC.covariance(K_profiled, residual);              % Gauss-Newton approximation only: this is
        end                                                                 % NOT the Ramsay Section 2.6 sandwich


        function [beta, H] = run_outer(obj, beta0, lb, ub)              % One outer solve from one starting point
            if PC.available('fmincon') && ~strcmpi(obj.settings.solver, 'builtin')
                options = optimoptions('fmincon', 'Display', 'none', ...
                                       'Algorithm', 'interior-point', ...
                                       'SpecifyObjectiveGradient', true, ...
                                       'MaxIterations', 500, ...
                                       'MaxFunctionEvaluations', 5000, ...
                                       'OptimalityTolerance', 1e-10, ...
                                       'StepTolerance', 1e-12);
                [beta, H] = fmincon(@(b) obj.outer_with_gradient(b), beta0, ...
                                    [], [], [], [], lb, ub, [], options);
                beta = reshape(beta, 1, []);
            else
                [beta, H] = obj.builtin_outer(beta0, lb, ub);               % fallback: no Optimization Toolbox
            end
        end


        function [beta, H] = builtin_outer(obj, beta0, lb, ub)          % Fallback: projected Gauss-Newton on H
            beta = reshape(beta0, 1, []);
            H = obj.outer(beta);
            for iter = 1:obj.settings.niter
                current = obj.coefficients();
                [H, K_profiled, residual] = obj.outer(beta);
                candidate = obj.bounded_qp(K_profiled, K_profiled*reshape(beta, [], 1) - residual);

                best_H = H;                                                 % keep the BEST step, not the first that
                best_beta = beta;                                           % improves
                step = 1;
                for trial = 1:obj.settings.nsearch
                    proposal = (1-step) * beta + step * candidate;
                    obj.set_coefficients(current);
                    trial_H = obj.outer(proposal);
                    if trial_H < best_H, best_H = trial_H; best_beta = proposal; end
                    step = step / 2;
                end

                obj.set_coefficients(current);
                if eucl_rel(beta, best_beta) < obj.settings.tol
                    beta = best_beta; H = best_H;
                    break
                end
                beta = best_beta;
                H = best_H;
            end
            beta = min(max(beta, lb), ub);
        end


        function starts = random_starts(obj, n)                         % Log-uniform starts around the current estimate
            lb = reshape(obj.settings.lb, 1, []);
            ub = reshape(obj.settings.ub, 1, []);
            low = max(lb, obj.beta_pc / 10);
            high = min(ub, obj.beta_pc * 10);
            low = max(low, 1e-12);
            high = max(high, low * 10);
            starts = exp(log(low) + rand(n, obj.P) .* (log(high) - log(low)));
        end


        function opt = bounded_qp(obj, design, response)                % Bounded least squares with the EGM prior
            prior = obj.settings.prior;
            lb = reshape(obj.settings.lb, 1, []);
            ub = reshape(obj.settings.ub, 1, []);

            H = design' * design + prior.prec;
            H = (H + H') / 2;
            f = -design' * response - prior.prec * prior.mean;

            A = kron([-1; 1], eye(length(lb)));
            b = [-lb ub]';
            try
                opt = mpcActiveSetSolver(H, f, A, b, zeros(0, length(lb)), zeros(0, 1), ...
                                         false(2*length(lb), 1), Optimization.options_mpcActiveSet)';
            catch ME
                disp(ME)
                opt = mpcActiveSetSolver(nearestSPD(H), f, A, b, zeros(0, length(lb)), zeros(0, 1), ...
                                         false(2*length(lb), 1), Optimization.options_mpcActiveSet)';
            end
            if any(~isfinite(opt)), opt = obj.beta_pc; end
            opt = min(max(opt, lb), ub);
        end


        function estimate_variances(obj)                                % Update measurement error variances
            for k = 1:obj.L
                state = obj.observed(k);
                predicted = repmat(obj.B{state} * obj.delta{state}, 1, 1, obj.N);
                design = [ones(obj.N*obj.T, 1) flatten(predicted).^2];
                response = flatten(predicted - obj.data.traces(:, k, :)).^2;

                coefficients = lsqnonneg(design, response)';
                if sum(coefficients) == 0, coefficients(1) = mean(response); end
                coefficients = Optimization.noise_parameters(coefficients, predicted, obj.data.traces(:, k, :));
                if sum(coefficients) == 0, coefficients(1) = mean(response); end

                obj.sigma2(k) = coefficients(1);
                obj.tau2(k) = coefficients(2);
                obj.variances_pc(:, k, :) = reshape(design * coefficients', obj.T, 1, obj.N);
            end
        end


        function save_estimates(obj)                                    % Extract results
            c = obj.coefficients();
            [obj.smoothed_pc, obj.dsmoothed_pc] = deal(zeros(obj.T, obj.K));
            [obj.smoothed_fine, obj.dsmoothed_fine] = deal(zeros(obj.data.T_fine, obj.K));

            init = zeros(1, obj.K);
            for k = 1:obj.K
                obj.smoothed_pc(:, k) = obj.B{k} * c(obj.cols{k});
                obj.dsmoothed_pc(:, k) = obj.dB{k} * c(obj.cols{k});
                obj.smoothed_fine(:, k) = obj.B_fine{k} * c(obj.cols{k});
                obj.dsmoothed_fine(:, k) = obj.dB_fine{k} * c(obj.cols{k});
                init(k) = obj.B{k}(1, :) * c(obj.cols{k});
            end

            if obj.settings.positive
                obj.smoothed_pc = max(obj.smoothed_pc, 1e-12);
                obj.smoothed_fine = max(obj.smoothed_fine, 1e-12);
                init = max(init, 1e-12);
            end

            [X, dX] = obj.states(c);
            Xe = X; if obj.settings.positive, Xe = max(X, 1e-12); end
            obj.discrepancy = dX - obj.system.rhs(Xe, obj.t_ode, obj.beta_pc);

            obj.data.init = init;                                           % store results
            obj.data.beta_pc = obj.beta_pc;
            obj.data.beta_iterations = obj.beta_iterations;
            obj.data.varbeta = obj.varbeta;
            obj.data.smoothed_pc = obj.smoothed_pc;
            obj.data.dsmoothed_pc = obj.dsmoothed_pc;
            obj.data.smoothed_fine = obj.smoothed_fine;
            obj.data.dsmoothed_fine = obj.dsmoothed_fine;
            obj.data.t_ode = obj.t_ode;
            obj.data.discrepancy = obj.discrepancy;
            obj.data.lambda_pc = obj.lambda;
            obj.data.criteria_pc = [obj.H_value obj.J_value];
            obj.data.variances_pc = obj.variances_pc;
            obj.data.delta_pc = obj.delta;
        end
    end


    methods (Static)
        function yes = available(name)                                  % Is an Optimization Toolbox solver present?
            yes = exist(name, 'file') == 2 || exist(name, 'builtin') == 5 || exist(name, 'file') == 6;
        end


        function w = trapezoid(t)                                       % Composite trapezoidal quadrature weights
            t = reshape(t, 1, []);
            w = zeros(size(t));
            w(1) = (t(2) - t(1)) / 2;
            w(end) = (t(end) - t(end-1)) / 2;
            w(2:end-1) = (t(3:end) - t(1:end-2)) / 2;
        end


        function x = normal_solve(J, rhs)                               % Regularized Gauss-Newton normal equations
            A = J' * J;
            A = (A + A') / 2;
            ridge = 1e-10 * max(1e-30, trace(A) / size(A, 1));
            x = PC.solve(A + ridge * eye(size(A)), J' * rhs, zeros(size(A, 1), size(rhs, 2)));
        end


        function x = solve(A, b, fallback)                              % Linear solve with a pseudoinverse fallback
            try
                x = A \ b;
            catch ME
                disp(ME)
                x = pinv(full(A)) * b;
            end
            if any(~isfinite(x(:))), x = fallback; end
        end


        function V = covariance(K_profiled, residual)                   % Gauss-Newton covariance approximation
            degrees = max(1, numel(residual) - size(K_profiled, 2));
            A = K_profiled' * K_profiled;
            A = (A + A') / 2;
            V = sum(residual.^2) / degrees * tryinv(A + 1e-12*eye(size(A)));
        end
    end
end
