%% Setup system ------------------------------------------------------------                
clearvars
                                                                            % generate ODE system object
system = System('model_repressilator.txt', 'auxiliary_repressilator.txt', ...
                 FixedParameters=["DNAT" "kf" "Kd" "m1" "p1"]);
save('system_repressilator.mat', 'system')
load('system_repressilator.mat')

                                                                            % setup generator
generator = Generator(system, N=1, t=0:5:100, error_std=.05, D_mult=eps, observed=1:3);

nseeds = 100;                                                               % number of repetitions
% error_sds = [.1 .2 .3];                                                     % parameter initialization variabilities
error_sds = [.25 .5 1];
[estimates_PC, estimates_MAGI, estimates_GP] = deal(zeros(nseeds, system.P));% allocate parameter estimates, ICs,
[ICs_PC, ICs_MAGI, ICs_GP] = deal(zeros(nseeds, system.K));                  % computation times, and model fits
[times_PC, times_MAGI, times_GP] = deal(zeros(nseeds, 2));
[fits_PC, fits_MAGI, fits_GP] = deal(zeros(nseeds, 1));

for error_idx = 1:length(error_sds)
    for seed = 1:nseeds
        rng(seed)
        [data, ground_truth] = generator.generate();                        % generate data
        
                                                                            % setup estimator
        estimator = EGM(system, data, Knots=10:10:90, InitialConditions=[10 20 30 .9 .9 .9 .2 .2 .2], Sigma=.1);
        
        error = exp(normrnd(0, error_sds(error_idx), 1, system.P));
        beta_init = error .* system.k0' %#ok<NOPTS>
        
        try
            disp('PC')
            out1 = estimator.estimate_PC(beta_init, [10 20 30 .9 .9 .9 .2 .2 .2]);                           % estimate using PC

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
            out2 = estimator.estimate_MAGI(beta_init, [10 20 30 .9 .9 .9 .2 .2 .2]);  % estimate using MAGI

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
            out3 = estimator.estimate_GP(beta_init, [10 20 30 .9 .9 .9 .2 .2 .2]);  % estimate using GP

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

save('simulation/compare_methods_repressilator.mat')                                   % store results
