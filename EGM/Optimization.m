classdef Optimization < handle                                          % General optimization object

    properties (Constant)                                                   % initialize once for efficiency
        options_mpcActiveSet = struct('DataType', 'double', 'MaxIterations', 200, 'ConstraintTolerance', 1e-6, ...
                                      'UseHessianAsInput', true, 'IntegrityChecks', false);
        options_interiorpoint = optimoptions('fmincon', 'Display', 'off', 'StepTolerance', 1e-8, ...
                                             'FiniteDifferenceType', 'central');
        options_interiorpoint_hightol = optimoptions('fmincon', 'Display', 'off', 'StepTolerance', 1e-4, ...
                                                     'FiniteDifferenceType', 'central');
        options_initialization = optimoptions('fmincon', 'Display', 'off', 'StepTolerance', 1e-4, ...
                                              'MaxIterations', 50);
    end


    methods (Static)                                                    % GLS using quadratic programming for gradient matching
        function [opt, SS] = QPGLS(design, response, covariance, lb, ub, prior)
            if nargin < 6, prior = struct('mean', 0, 'prec', 0); end

            P = size(design, 2);                                            % number of parameters
            if nargin <= 4, lb = zeros(1, P); ub = Inf(1, P); end
            precision = tryinv(covariance);

            Y = flatten(response);
            X = design;
            H = X' * precision * X + prior.prec;                            % quadratic programming problem
            H = (H+H')/2;                                                   % force symmetry
            f = -X' * precision * Y - prior.prec * prior.mean;
            if any(f ~= real(f), 'all') || any(f ~= real(f))
                disp(1)
            end

            A = kron([-1; 1], eye(length(lb)));                             % bounds to inequality constraints
            b = [-lb ub]';
            Aeq = zeros(0, length(lb));                                     % empty equality constraints
            beq = zeros(0, 1);                                              % solve using active-set algorithm
            try
                opt = mpcActiveSetSolver(H, f, A, b, Aeq, beq, false(2*length(lb), 1), Optimization.options_mpcActiveSet)';
            catch ME
                H = nearestSPD(H);                                          % retry if H is not SPD
                opt = mpcActiveSetSolver(H, f, A, b, Aeq, beq, false(2*length(lb), 1), Optimization.options_mpcActiveSet)';
            end

            SS = (Y - X*opt')' * precision * (Y - X*opt');
        end


        function coef = noise_parameters(start, pred, measurements)     % Particular optimizer for noise variance parameters
            ws = warning('off', 'MATLAB:nearlySingularMatrix');
            assert(numel(start) == 2)

            pred = flatten(pred);
            measurements = flatten(measurements);

            L = @(p) sum(log(p(1) + p(2).*pred.^2) + (measurements - pred).^2 ./ (p(1) + p(2).*pred.^2));
            coef = fmincon(L, start, [], [], [], [], [0 0], [Inf Inf], [], Optimization.options_interiorpoint);

            warning(ws)
        end

                                                                         % (Multistarted) numerical least squares
        function [beta, value] = least_squares(SS, lb, ub, nstart, init, ~) % (in particular for trajectory matching)
            ws = warning('off', 'MATLAB:nearlySingularMatrix');

            options = Optimization.options_interiorpoint;                   % switch multistarts and options
            if nargin == 5 && ~isempty(init)
                nstart = 1;
            elseif nargin < 5
                options = Optimization.options_initialization;
            end

            if nargin == 6, options = Optimization.options_initialization; end

            value = Inf;
            for start = 1:nstart
                if nargin == 5 && ~isempty(init)                            % switch initial point
                    loginit = log(init);
                else                                                        % uniformly sample in log parameter space
                    loginit = rand(1, numel(lb)) .* (log(ub) - log(lb)) + log(lb);
                end

                [opt_new, value_new] = fmincon(@(logbeta) SS(exp(logbeta)), loginit, [], [], [], [], ...
                                               log(lb), log(ub), [], options);
                if value_new < value, value = value_new; opt = opt_new; end
            end
            beta = exp(opt);                                                % transform back to normal scale

            warning(ws)
        end
    end
end


