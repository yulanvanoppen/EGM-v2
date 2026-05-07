classdef TM < handle                                                    % Trajectory matching estimator
    properties (SetAccess = private)
        data                                                                % data aggregate struct
        system                                                              % nested object controlling the ODE system
        settings                                                            % hyperparameters and user input
        
        T                                                                   % number of time points
        L                                                                   % number of observed states
        N                                                                   % population size
        
        beta_fs                                                             % cell-specific parameter estimates
        varbeta                                                             % uncertainty estimates of beta
        sigma2                                                              % additive measurement error variance 
        tau2                                                                % multiplicative measurement error variance 
        variances_fs                                                        % measurement error variances along time grid
        
        convergence_steps                                                   % relative iteration steps
        not_converged                                                       % indices of cells that have not yet converged
        
        fitted_fs                                                           % predicted states and gradients
        dfitted_fs
    end
    
    
    methods
        function obj = TM(data, system, settings, beta_init, x_init) % Constructor
            obj.data = data;                                                % experimental and smoothing data
            obj.data.init = x_init;
            obj.system = system;                                            % ODE system
            obj.settings = settings;                                        % hyperparameters
            
            [obj.T, obj.L, obj.N] = size(data.traces);                      % extract from trajectory dimensions
            
            obj.beta_fs = beta_init;
            obj.varbeta = repmat(eye(system.P), 1, 1, obj.N);
            obj.variances_fs = data.traces.^2 + .1*mean(data.traces).^2;    % initialize mostly multiplicative
        end
        
        
        function output = optimize(obj, estimate_x0)                    % Main optimization function
            if nargin == 1, estimate_x0 = false; end

            for iter = 1:obj.settings.niter
                fprintf('%d ', iter)

                if estimate_x0 && iter == 1
                    obj.init_x0();
                end

                beta_old = obj.beta_fs;
                
                obj.update_parameters(estimate_x0);                         % update (cell-specific) parameter estimates
                % obj.update_parameters(false);                               % update (cell-specific) parameter estimates

                obj.estimate_variances();                                   % update variance estimates

                if ~mod(iter, 10), fprintf('(%.3e)\n', eucl_rel(beta_old, obj.beta_fs)), end

                if eucl_rel(beta_old, obj.beta_fs) < obj.settings.tol || iter == obj.settings.niter
                    fprintf('(%.3e)\n', eucl_rel(beta_old, obj.beta_fs))
                    break
                end
            end
            
            obj.save_estimates();                                           % save estimates and fitted trajectories
            output = obj.data;                                              % return data appended with FS results
        end


        function init_x0(obj)
            center = obj.data.init;
            center(obj.data.observed) = obj.data.traces(1, :);
            multipliers = 4.^(-1:.5:1);
            uncertain = find(~ismember(1:obj.system.K, obj.data.observed));
            switch string(obj.system.name)
                case "Epo"
                    center([2 3]) = center([4 6]);
                    uncertain = [1 3];
                case "STAT5"
                    uncertain = [3 4];
                case "repressilator"
                    uncertain = [4 7];
                case "generalizedLV"
                    uncertain = 1;
                case "bifunctional_TCS"
                    uncertain = 4;
            end
            
            conditions = repmat(center', [1 length(multipliers) * ones(1, length(uncertain))]);
            switch length(uncertain)
                case 1
                    conditions(uncertain, :) = center(uncertain) * multipliers;
                    conditions = conditions';
                    if string(obj.system.name) == "generalizedLV"
                        conditions(:, 2) = conditions(:, 1);
                    end
                case 2
                    mult = ones(size(conditions));
                    mult(uncertain(1), :, :) = repmat(multipliers, 1, 1, length(multipliers));
                    mult(uncertain(2), :, :) = repmat(reshape(multipliers, 1, 1, []), 1, length(multipliers), 1);
                    conditions = reshape(conditions .* mult, length(center), [])';
                    if string(obj.system.name) == "repressilator"
                        conditions(:, [5 6]) = repmat(conditions(:, 4), 1, 2);
                        conditions(:, [8 9]) = repmat(conditions(:, 7), 1, 2);
                    end
            end

            if string(obj.system.name) == "STAT5"
                conditions(:, 4) = min(conditions(:, 3), conditions(:, 4));
            end

            errors = inf(1, size(conditions, 1));
            betas = zeros(size(conditions, 1), obj.system.P);
            x0s = zeros(size(conditions, 1), obj.system.K);

            SS = @(beta_x0) obj.squares_sum(beta_x0(1:obj.system.P), obj.data.traces, obj.variances_fs, beta_x0(obj.system.P+1:end));
            for cond = 1:size(conditions, 1)
                try
                    obj.data.init = conditions(cond, :);
                    [beta_x0, errors(cond)] = Optimization.least_squares(SS, [obj.settings.lb, eps - 100*ones(1, obj.system.K) * ~obj.settings.positive], ...
                                                                                [obj.settings.ub, 10000*ones(1, obj.system.K)], ...
                                                                                1, [obj.beta_fs obj.data.init], 'high_tolerance');
                    betas(cond, :) = beta_x0(1:obj.system.P);
                    x0s(cond, :) = beta_x0(obj.system.P+1:end);
                catch ME
                    disp(ME)
                end
            end

            [~, cond] = min(errors(:));
            obj.data.init = conditions(cond, :);
            obj.beta_fs = betas(cond, :);
        end
            
        
        function update_parameters(obj, estimate_x0)                    % Update (cell-specific) parameter estimates
            if estimate_x0
                % % selection = [0 1 0 0 0 0];
                % selection = ones(1, 6);
                % selected = @(x0) selection .* x0 + (1-selection) .* obj.system.x0 + 1e-8;
                % SS = @(beta_x0) obj.squares_sum(beta_x0(1:obj.system.P), obj.data.traces, obj.variances_fs, selected(beta_x0(obj.system.P+1:end)));

                if string(obj.system.name) == "Epo"
                    P = obj.system.P;
                    SS = @(beta_x0) obj.squares_sum(beta_x0(1:P), obj.data.traces, obj.variances_fs, ...
                                                    beta_x0([P+1:P+3 P+2 P+5 P+3])) + sum(beta_x0([P+4 P+6]).^2);
                else
                    SS = @(beta_x0) obj.squares_sum(beta_x0(1:obj.system.P), obj.data.traces, obj.variances_fs, beta_x0(obj.system.P+1:end));
                end

                beta_x0 = Optimization.least_squares(SS, [obj.settings.lb, eps - 100*ones(1, obj.system.K) * ~obj.settings.positive], ...
                                                    [obj.settings.ub, 10000*ones(1, obj.system.K)], 1, [obj.beta_fs obj.data.init]);
                obj.beta_fs = beta_x0(1:obj.system.P);
                if string(obj.system.name) == "Epo"
                    obj.data.init = beta_x0([P+1:P+3 P+2 P+5 P+3]);
                else
                    obj.data.init = beta_x0(obj.system.P+1:end);
                end
    
                obj.fitted_fs = obj.system.integrate(obj.beta_fs, obj.data);    % compute predicted trajectories
                if obj.settings.positive
                    obj.fitted_fs = max(1e-12, obj.fitted_fs);                  % force positive
                end

            else
                SS = @(beta) obj.squares_sum(beta, obj.data.traces, obj.variances_fs);
                obj.beta_fs = Optimization.least_squares(SS, obj.settings.lb, obj.settings.ub, 1, obj.beta_fs);
    
                obj.fitted_fs = obj.system.integrate(obj.beta_fs, obj.data);    % compute predicted trajectories
                if obj.settings.positive
                    obj.fitted_fs = max(1e-12, obj.fitted_fs);                  % force positive
                end
            end
        end
        
        
        function ss = squares_sum(obj, beta, y, variances, x0)          % Weighted sum of squared differences
            int_data = obj.data;
            if nargin == 5, int_data.init = x0; end

            ss = Inf;
            try                                                 
                solution = obj.system.integrate(beta, int_data);            % compute fitted trajectories
                solution = solution(:, obj.data.observed);
                
                ss = sum((solution - y).^2 ./ variances, 'all');            % sum of squares on observed states
            catch ME
                disp(ME)
            end
        end
        
        
        function estimate_variances(obj)                                % Update measurement error variances
            for k = 1:obj.L
                predicted = obj.fitted_fs(:, obj.data.observed(k), :);      % fitted trajectories
                design = [ones(obj.N*obj.T, 1) flatten(predicted).^2];      % columns for additive and multiplicative noise
                response = flatten(predicted - obj.data.traces(:, k, :)).^2;
                
                coefficients = lsqnonneg(design, response)';                % initialize nonzero LS estimates for noise parameters
                if sum(coefficients) == 0, coefficients(1) = mean(response); end
                                                                            % optimize further iteratively
                coefficients = Optimization.noise_parameters(coefficients, predicted, obj.data.traces(:, k, :));
                if sum(coefficients) == 0, coefficients(1) = mean(response); end
                
                obj.sigma2(k) = coefficients(1);                            % store optimum and compute variances
                obj.tau2(k) = coefficients(2);
                
                obj.variances_fs(:, k, :) = reshape(design * coefficients', obj.T, 1, obj.N);
            end
        end
        
        
        function save_estimates(obj)                                    % Extract results 
            obj.data.init
            obj.data.beta_fs = obj.beta_fs;                                 % store results
            obj.data.fitted_fs = obj.fitted_fs;
            obj.data.dfitted_fs = obj.dfitted_fs;
            obj.data.varbeta = obj.varbeta;
        end
    end
end








