classdef GM < handle                                                    % Gradient matching estimator
    properties (SetAccess = private)
        data                                                                % general data and output container
        system                                                              
        settings                                                            % hyperparameters and user input
        
        T                                                                   % number of data time points
        T_gm                                                                % number of first stage time points
        T_fine                                                              % number of time points for smooth plots
        L                                                                   % system dimension
        
        B
        dB
        
        C
        
        smoothed_filtered
        dsmoothed_dfiltered
        
        observed
        hidden

        b
        g
        fdiff
        variances_XdX
    end
    
    
    methods
        function obj = GM(system, data, settings)                       % Constructor
            obj.data = data;                                                % store data, ODE system, and hyperparameters
            obj.system = system;                                            
            obj.settings = settings;
            [obj.T, obj.L] = size(obj.data.traces);                         % extract trajectory dimensions
            obj.T_gm = length(settings.t_gm);
            
            obj.B = blkdiag(obj.data.basis{:});
            obj.dB = blkdiag(obj.data.dbasis{:});
            
            obj.C = zeros(obj.L, obj.system.K);
            obj.C(:, obj.data.observed) = eye(obj.L);
            
            obj.smoothed_filtered = zeros(obj.T, obj.system.K);
            obj.dsmoothed_dfiltered = zeros(obj.T, obj.system.K);
            obj.smoothed_filtered(:, data.observed) = data.smoothed;
            obj.dsmoothed_dfiltered(:, data.observed) = data.dsmoothed;
            
            obj.observed = data.observed;
            obj.hidden = 1:obj.system.K;
            obj.hidden = obj.hidden(~ismember(1:obj.system.K, obj.observed));

            order = 3; framelen = 7;
            [obj.b, obj.g] = sgolay(order, framelen);

            [~, g5] = sgolay(order, 5);
            [~, g3] = sgolay(order-2, 3);

            t = data.t;
            dt = [t(2)-t(1), (t(3:end) - t(1:end-2))/2, t(end)-t(end-1)];
            diagonals = repmat(obj.g(:, 2)', obj.T, 1);
            obj.fdiff = spdiags(diagonals, -(framelen-1)/2:(framelen-1)/2, obj.T, obj.T);

            obj.fdiff([1:3 end-2:end], :) = 0;

            obj.fdiff(3, 1:5) = g5(:, 2)';
            obj.fdiff(end-2, end-4:end) = g5(:, 2)';

            obj.fdiff(2, 1:3) = g3(:, 2)';
            obj.fdiff(end-1, end-2:end) = g3(:, 2)';

            obj.fdiff(1, 1:2) = [-1 1];
            obj.fdiff(end, end-1:end) = [-1 1];

            obj.fdiff = obj.fdiff ./ dt;
        end
        
        
        function [beta, SS, precision] = estimate(obj, beta, filtered)
            t = obj.data.t;
            
            filtered_dxbar = obj.fdiff * filtered.xbar';

            obj.smoothed_filtered(:, obj.hidden) = filtered.xbar(obj.hidden, :)';
            obj.dsmoothed_dfiltered(:, obj.hidden) = filtered_dxbar(:, obj.hidden);

            G_smoothed_filtered = obj.system.g(obj.smoothed_filtered(2:end-1, :), t(2:end-1));
            H_smoothed_filtered = obj.system.h(obj.smoothed_filtered(2:end-1, :), t(2:end-1));

            design = G_smoothed_filtered;
            const = H_smoothed_filtered;
            response = obj.dsmoothed_dfiltered(2:end-1, :) - const;

            V = obj.covariances(beta, filtered);

            [beta, SS] = Optimization.QPGLS(design, response, nearestSPD(V), ...
                                            obj.settings.lb, obj.settings.ub, obj.settings.prior);

            precision = obj.uncertainties(V);
        end
        
        
        function variances = covariances(obj, beta, filtered)
            K = obj.system.K;
            KT = K*obj.T;
            LT = obj.L*obj.T;
            LT_data = obj.L*obj.data.T_data;
            
            ind_hid = flatten((1:obj.T)'+(obj.T*(obj.hidden-1)));           % hidden/observed state/time indices
            ind_obs = flatten((1:obj.T)'+(obj.T*(obj.observed-1)));

            indices_t = 2:obj.T-1;                                          % time indices without interval ends
            indices_tk = reshape(indices_t' + obj.T*(0:K-1), 1, []); % across states
            indices_2tk = reshape(indices_tk' + [0 KT], 1, []); % across state and gradient components

            dRHS_mixed = obj.system.df(obj.smoothed_filtered, obj.data.t, beta);

            fdiff_cell = cell(1, K);
            [fdiff_cell{:}] = deal(obj.fdiff);
            fdiff_full = blkdiag(fdiff_cell{:});

            delta_X_dX_y = [eye(KT)       zeros(KT, LT_data);
                            fdiff_full    zeros(KT, LT_data);
                            zeros(LT_data, KT)  eye(LT_data)];

            covar = delta_X_dX_y * filtered.covar * delta_X_dX_y';


            lincomb_data = blkdiag(obj.data.lincomb_data{:});
            
            delta_X_dX = zeros(2*KT, 2*KT+LT_data);
            delta_X_dX(ind_hid, ind_hid) = eye(KT-LT);
            delta_X_dX(ind_obs, end-LT_data+1:end) = obj.B * lincomb_data;
            delta_X_dX(ind_hid+KT, ind_hid+KT) = eye(KT-LT);
            delta_X_dX(ind_obs+KT, end-LT_data+1:end) = obj.dB * lincomb_data;

            obj.variances_XdX = delta_X_dX * covar * delta_X_dX';

            dRHS = zeros(KT);                                               % variance of X, dX, AND dX - G(X)beta - H(X)
            for j = 1:obj.T
                dRHS(j + obj.T*(0:K-1), j + obj.T*(0:K-1)) = dRHS_mixed(j + obj.T*(0:K-1), :);
            end
            delta_residuals = [-dRHS eye(KT)];

            V = delta_residuals(indices_tk, indices_2tk) * obj.variances_XdX(indices_2tk, indices_2tk) ...
                                                         * delta_residuals(indices_tk, indices_2tk)';

            regulator = max(1e-12, max(abs(obj.smoothed_filtered(:, :)) / 1e6, [], 1));
            regulator = reshape(repmat(regulator, obj.T-2, 1), 1, []);
            variances = nearestSPD(V + diag(regulator));
        end


        function covar = uncertainties(obj, V)                          % Parameter uncertainties
            K = obj.system.K;                                               % abbreviate
            KT = K*obj.T;
            
            indices_t = 2:obj.T-1;                                          % time indices without interval ends
            indices_tk = reshape(indices_t' + obj.T*(0:K-1), 1, []);        % across states
            indices_2tk = reshape(indices_tk' + [0 KT], 1, []);             % across state and gradient components

            g_all = obj.system.g(obj.smoothed_filtered, obj.data.t);        % compute g(.), h(.), and their Jacobians for each cell
            dg_all = obj.system.dg(obj.smoothed_filtered, obj.data.t);
            h_all = obj.system.h(obj.smoothed_filtered, obj.data.t);
            dh_all = obj.system.dh(obj.smoothed_filtered, obj.data.t);

            dX = flatten(obj.dsmoothed_dfiltered(indices_t, :));            % left-hand side
            H = flatten(h_all(indices_t, :));                               % constant part wrt parameters
            G = g_all(indices_tk, :);                                       % linear part wrt parameters
            
            Vinv = tryinv(V);                                               % inverse residual covariance matrix estimate
            
            Th = G' * Vinv * G;                                             % construct d[beta]/d[dX] and d[beta]/d[X]
            Thinv = tryinv(Th);                                             % cf. section S6 of https://doi.org/10.1093/bioinformatics/btaf154
            dbeta_ddX = Thinv * G' * Vinv;                                  % partial with respect to dX

            Xi = G' * Vinv * (dX - H);
            Thinv_Xi = Thinv * Xi;
            Pi = zeros(obj.system.P, obj.T, K);
            Psi = zeros(obj.system.P, obj.T, K);
            for k = 1:K
                for j = 2:obj.T-1
                    dg_dxk = dg_all(:, :, j, k);
                    Vinv_j = Vinv(j-1 + (obj.T-2)*(0:K-1), :);

                    Pi(:, j, k) = (dg_dxk' * Vinv_j * G + G' * Vinv_j' * dg_dxk) * Thinv_Xi;
                    Psi(:, j, k) = dg_dxk' * Vinv_j * (dX-H) - G' * Vinv_j' * dh_all(j + obj.T*(0:K-1), k);
                end
            end                                                             % partial with respect to X
            dbeta_dX = Thinv * reshape(Psi(:, indices_t, :) - Pi(:, indices_t, :), obj.system.P, []);
            
                                                                            % delta method approximation
            covar = [dbeta_dX dbeta_ddX] * obj.variances_XdX(indices_2tk, indices_2tk) * [dbeta_dX dbeta_ddX]';
        end
    end
end