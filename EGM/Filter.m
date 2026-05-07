classdef Filter < handle
    properties (SetAccess = private)
        data                                                                % general data and output container
        system                                                              
        settings                                                            % hyperparameters and user input
        
        T                                                                   % number of data time points
        T_gm                                                                % number of first stage time points
        T_fine                                                              % number of time points for smooth plots
        L                                                                   % system dimension
        N                                                                   % number of cells
        
        sigma                                                               % process noise
        x_init                                                              % initial condition prior mean
        P_init                                                              % initial condition prior variance
        P_CV                                                                % initial condition prior CV
        
        C                                                                   % observation function
    end
    
    
    methods
        function obj = Filter(system, data, settings)                   % Constructor
            obj.data = data;                                                % store data, ODE system, and hyperparameters
            obj.system = system;                                            
            obj.settings = settings;
            [obj.T, obj.L, obj.N] = size(obj.data.traces);                  % extract trajectory dimensions
            obj.T_gm = length(settings.t_gm);
            
            obj.sigma = obj.settings.sigma;
            obj.x_init = obj.settings.x_init;
            regularizer = .01 * mean(obj.data.traces, 'all');               % initialize regularized IC prior covariances
            obj.P_init = diag((obj.settings.P_CV .* obj.x_init).^2 + regularizer^2);
            obj.P_CV = obj.settings.P_CV;
            
            obj.C = zeros(obj.L, obj.system.K);                             % observed state selector matrix
            obj.C(:, obj.data.observed) = eye(obj.L);                       % (extensible if necessary)
        end


        function update_ICmean(obj, x)                                  % Set initial condition prior mean
            obj.x_init = x;
        end

        
        function filtered = EKS(obj, beta)
            K = obj.system.K;                                               % numbers of states and time points
            KT = K * obj.T;
            LT = obj.L * obj.T;
            y = obj.data.traces';                                           % measurements
            
            [xbar, xbar_pre, xbar_sm, sd, sd_sm] = deal(zeros(K, obj.T));   % allocate
            [P, P_pre, P_sm, Phi, Q] = deal(zeros(K, K, obj.T));

            [xbar(:, 1), xbar_pre(:, 1)] = deal(obj.x_init);                % initialize from prior
            [P(:, :, 1), P_pre(:, :, 1)] = deal(obj.P_init);
            sd(:, 1) = sqrt(diag(P(:, :, 1)));                              % record in parallel for plotting


            %% EKF ---------------------------------------------------------
                                                                            % Process first data point
            [KG, xbar(:, 1), P(:, :, 1), sd(:, 1)] = obj.Kalman_gain(1, y, xbar_pre, P_pre);
            covar = obj.update_covariance_EKF(1, KG, P(:, :, 1));
            
            for n = 2:obj.T                                                 % solve original and linear matrix ODEs
                [tout, out] = obj.integrate_augmented(beta, xbar(:, n-1), ...
                                                      [obj.data.t_data(n-1) obj.data.t_data(n)]);
                                                       
                [Q_n, Phi_n] = obj.cumulative_noise(tout, out);             % approximate cumulative noise integral
                Phi(:, :, n) = Phi_n;
                Q(:, :, n) = Q_n;
                
                xbar_pre(:, n) = out(end, 1:K);                             % update prior estimates at current time point
                P_pre(:, :, n) = Phi_n * P(:, :, n-1) * Phi_n' + Q_n;
                                                                            % apply Kalman gain and expand covariance matrix
                [KG, xbar(:, n), P(:, :, n), sd(:, n)] = obj.Kalman_gain(n, y, xbar_pre, P_pre);
                covar = obj.update_covariance_EKF(n, KG, P(:, :, n), covar, Phi_n);
            end

            %% EKS ---------------------------------------------------------
            xbar_sm(:, obj.T) = xbar(:, obj.T);                             % initialize from final EKF estimates
            P_sm(:, :, obj.T) = P(:, :, obj.T);
            sd_sm(:, obj.T) = sd(:, obj.T);
            covar = covar([KT+1:KT+LT 1:KT], [KT+1:KT+LT 1:KT]);

            for n = 1:obj.T-1
                T_n = obj.T - n;                                            % proceed backward
                                                                            %  apply 'reverse gain'
                G_n = P(:, :, T_n) * Phi(:, :, T_n+1)' * tryinv(P_pre(:, :, T_n+1));
                xbar_sm(:, T_n) = xbar(:, T_n) + G_n * (xbar_sm(:, T_n+1) - xbar_pre(:, T_n+1)); 
                P_sm(:, :, T_n) = P(:, :, T_n) + G_n * (P_sm(:, :, T_n+1) - P_pre(:, :, T_n+1)) * G_n';
                sd_sm(:, T_n) = sqrt(diag(P_sm(:, :, T_n)));
                                                                            
                covar = obj.update_covariance_EKS(n, G_n, P_sm(:, :, T_n), covar, Phi(:, :, T_n+1));
            end

            reordered_L = reshape((1:obj.L) + obj.L*(0:obj.T-1)', 1, []);   % group time points instead of states
            reordered_K = reshape(obj.T*(obj.L+K) + (1:K) + K*(obj.T-2:-1:-1)', 1, []); % + revert smoothed states
            covar = covar([reordered_K reordered_L], [reordered_K reordered_L]);

                                                                            % save results to output struct
            filtered = struct('xbar', xbar_sm, 'P', P_sm, 'sd', sd_sm, 'covar', covar);
        end
        
                                                                        % Compute and apply Kalman gain
        function [KG, xbar, P, sd] = Kalman_gain(obj, n, y, xbar_pre, P_pre)
            K = obj.system.K;
            I = eye(K);
            
            y = y(:, n);
            xbar_pre = xbar_pre(:, n);                                      % prior before time point tn
            P_pre = P_pre(:, :, n);
            Sigma = diag(obj.data.variances_sm(n, :));                      % measurement errors variances
            
            KG = P_pre * obj.C' / (obj.C * P_pre * obj.C' + Sigma);         % Kalman gain
            
            xbar = xbar_pre(:, 1) + KG * (y - obj.C * xbar_pre);            % posterior mean after tn
            
            P = (I - KG*obj.C) * P_pre * (I - KG*obj.C)' + KG * Sigma * KG';% posterior covariance matrix after tn
            sd = sqrt(diag(P));
        end
        
        
        function covar = update_covariance_EKF(obj, n, KG, P, covar, Phi) % Expand covar matrix
            K = obj.system.K;
            TK = K * obj.T;
            
            if n == 1                                                       % allocate/initialize at first time point
                Sigma = diag(obj.data.variances_sm(1, :));
                covar = zeros(obj.T*(K+obj.L));
                covar(1:K, 1:K) = P;
                covar(1:K, TK+1:TK+obj.L) = KG * Sigma;
                covar(TK+(1:obj.L), 1:K) = covar(1:K, TK+(1:obj.L))';
                covar(TK+(1:obj.L), TK+(1:obj.L)) = Sigma;
            else
                Sigma = diag(obj.data.variances_sm(n, :));                  % measurement errors variances
                I = eye(K);
                
                factor = (I - KG*obj.C) * Phi;
                
                U = covar(1:K*(n-1), K*(n-2) + (1:K));
                covar(1:K*(n-1), K*(n-1) + (1:K)) = U * factor';            % update Uin, i=1,...,n-1
                covar(K*(n-1) + (1:K), 1:K*(n-1)) = covar(1:K*(n-1), K*(n-1) + (1:K))';

                covar(K*(n-1) + (1:K), K*(n-1) + (1:K)) = P;                % update Unn

                W = covar(K*(n-2) + (1:K), obj.T*K + (1:obj.L*(n-1)));      % update Win, i=1,...,n-1
                covar(K*(n-1) + (1:K), obj.T*K + (1:obj.L*(n-1))) = factor * W;
                covar(obj.T*K + (1:obj.L*(n-1)), K*(n-1) + (1:K)) = covar(K*(n-1) + (1:K), obj.T*K + (1:obj.L*(n-1)))';

                                                                            % update Wnn
                covar(K*(n-1) + (1:K), obj.T*K + obj.L*(n-1) + (1:obj.L)) = KG * Sigma;
                covar(obj.T*K + obj.L*(n-1) + (1:obj.L), K*(n-1) + (1:K)) = covar(K*(n-1) + (1:K), obj.T*K + obj.L*(n-1) + (1:obj.L))';
                
                covar(obj.T*K + obj.L*(n-1) + (1:obj.L), ...               % add measurement variance
                      obj.T*K + obj.L*(n-1) + (1:obj.L)) = Sigma;
            end
        end


        function covar = update_covariance_EKS(obj, n, G, P_sm, covar, Phi) % Expand covar matrix
            K = obj.system.K;
            existing = obj.T*(K+obj.L);
            ind_obs = 1:obj.T*obj.L;
            
            if n == 1                                                       % extend at first step (last time point)
                new = zeros(existing+(obj.T-1)*K);
                new(1:existing, 1:existing) = covar;
                covar = new;
            end

            W = covar(ind_obs, existing - K*(n+1) + (1:K));                 % update according to derivations
            S = covar(ind_obs, existing + K*(n-2) + (1:K));
            T_cov = covar(existing-K + (1:K*n), existing + K*(n-2) + (1:K));

            covar(ind_obs, existing + K*(n-1) + (1:K)) = W + S * G' - W * Phi' * G';
            covar(existing + K*(n-1) + (1:K), ind_obs) = covar(ind_obs, existing + K*(n-1) + (1:K))';
            covar(existing-K + (1:K*n), existing + K*(n-1) + (1:K)) = T_cov * G';
            covar(existing + K*(n-1) + (1:K), existing-K + (1:K*n)) = covar(existing-K + (1:K*n), existing + K*(n-1) + (1:K))';
            covar(existing + K*(n-1) + (1:K), existing + K*(n-1) + (1:K)) = P_sm;
        end
        
                                                                        % Integrate auxiliary IQM tools model
        function [tout, yout] = integrate_augmented(obj, beta, x0, trange)
            tout = linspace(trange(1), trange(2), 21);                      % 20 subintervals between time points for integration
            yout = obj.system.auxiliary(beta, x0', tout, 1E-4);
        end
        
        
        function [Q_it, Phi_it] = cumulative_noise(obj, tout, out)      % Integrate linear matrix ODE and cumulative noise
            K = obj.system.K;
            
            Phi = permute(reshape(out(:, K+1:end), [], K, K), [2 3 1]);     % reshape to square matrix output

            S_eval = obj.sigma.*out(:, 1:K)';                               % constant process noise function

            integrand = zeros(size(Phi));
            for tidx = 1:length(tout)                                       % precompute for efficiency
                Phiti_Sti = Phi(:, :, tidx) .* S_eval(:, tidx)';
                integrand(:, :, tidx) = Phiti_Sti * Phiti_Sti';
            end

            Q_it = reshape(trapz(tout, integrand, 3), K, K);                % (fast) trapezoidal integration
            Phi_it = reshape(Phi(:, :, end), K, K);
        end
        
    end
end