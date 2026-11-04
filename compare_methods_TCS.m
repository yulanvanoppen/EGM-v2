%% Setup system ------------------------------------------------------------
clearvars
                                                                            % generate ODE system object
system = System('model_TCS.txt', 'auxiliary_TCS.txt', FixedParameters=["k2" "k3" "k6" "k8"]);
save('system_TCS.mat', 'system')
load('system_TCS.mat')
                                                                            % setup generator
generator = Generator(system, N=1, t=[0 1 3 5:5:50], error_std=.05, D_mult=eps, observed=[3 5:6]);

nseeds = 100;                                                               % number of repetitions
% error_sds = [.1 .2 .3];                                                     % parameter initialization variabilities
error_sds = [.25 .5 1];
[estimates_PC, estimates_MAGI, estimates_GP] = deal(zeros(nseeds, system.P));% allocate parameter estimates, ICs,
[ICs_PC, ICs_MAGI, ICs_GP] = deal(zeros(nseeds, system.K));                  % computation times, and model fits
[times_PC, times_MAGI, times_GP] = deal(zeros(nseeds, 2));
[fits_PC, fits_MAGI, fits_GP] = deal(zeros(nseeds, 1));

for error_idx = 1:length(error_sds)
    for seed = 1:nseeds
        rng(10*seed+error_idx)
        [data, ground_truth] = generator.generate();                            % generate data
        
                                                                                % setup estimator
        estimator = EGM(system, data, Knots=[0 3 25], InitialConditions=[0 0 .2 2.5 0 0]+eps, ...
                        LB=.1*system.k0', UB=10*system.k0', ...                         % wider bounds
                        Prior=struct('mean', [1 1 1 1], 'prec', diag([0 0 0 0])), ...   % prior to prevent unbounded k7
                        Sigma=.1);
    
        error = exp(normrnd(0, error_sds(error_idx), 1, system.P));
        beta_init = error .* system.k0'
        
        try
            disp('PC')
            out1 = estimator.estimate_PC(beta_init, [0 0 .2 2.5 0 0]+eps);                           % estimate using PC

            estimates_PC(seed, :, error_idx) = out1.beta;                  % store parameter estimates and ICs
            ICs_PC(seed, :, error_idx) = out1.init;
            times_PC(seed, :, error_idx) = out1.time;                      % store computation times
            discrepancies_PC = (data.traces - out1.fitted(:, data.observed));% store model fits
            fits_PC(seed, error_idx) = mean(discrepancies_PC.^2 ./ ...
                                                ground_truth.original(:, data.observed), 'all');
        catch ME
            % disp(ME)                                                        % catch any errors, record failure
            estimates_PC(seed, :, error_idx) = 99*ones(size(system.k0'));
            times_PC(seed, :, error_idx) = [99 99];
            fits_PC(seed, error_idx) = 99;
            ICs_PC(seed, :, error_idx) = 99*ones(size(system.x0'));
        end

        try
            disp('MAGI')
            out2 = estimator.estimate_MAGI(beta_init, [0 0 .2 2.5 0 0]+eps);  % estimate using MAGI

            estimates_MAGI(seed, :, error_idx) = out2.beta;
            ICs_MAGI(seed, :, error_idx) = out2.init;
            times_MAGI(seed, :, error_idx) = out2.time;
            discrepancies_MAGI = (data.traces - out2.fitted(:, data.observed));
            fits_MAGI(seed, error_idx) = mean(discrepancies_MAGI.^2 ./ ...
                                                ground_truth.original(:, data.observed), 'all');
        catch ME
            % disp(ME)                                                        % catch any errors, record failure
            estimates_MAGI(seed, :, error_idx) = 99*ones(size(system.k0'));
            times_MAGI(seed, :, error_idx) = [99 99];
            fits_MAGI(seed, error_idx) = 99;
            ICs_MAGI(seed, :, error_idx) = 99*ones(size(system.x0'));
        end

        try
            disp('GP')
            out3 = estimator.estimate_GP(beta_init, [0 0 .2 2.5 0 0]+eps);  % estimate using GP

            estimates_GP(seed, :, error_idx) = out3.beta;
            ICs_GP(seed, :, error_idx) = out3.init;
            times_GP(seed, :, error_idx) = out3.time;
            discrepancies_GP = (data.traces - out3.fitted(:, data.observed));
            fits_GP(seed, error_idx) = mean(discrepancies_GP.^2 ./ ...
                                                ground_truth.original(:, data.observed), 'all');
        catch ME
            % disp(ME)                                                        % catch any errors, record failure
            estimates_GP(seed, :, error_idx) = 99*ones(size(system.k0'));
            times_GP(seed, :, error_idx) = [99 99];
            fits_GP(seed, error_idx) = 99;
            ICs_GP(seed, :, error_idx) = 99*ones(size(system.x0'));
        end
    end
end

save('simulation/compare_methods_TCS.mat')                                             % store results
