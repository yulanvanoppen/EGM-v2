%% Setup system ------------------------------------------------------------                
clearvars
                                                                            % generate ODE system object
% system = System('model_repressilator.txt', 'auxiliary_repressilator.txt', ...
%                  FixedParameters=["DNAT" "kf" "Kd" "m1" "p1"]);
% save('system_repressilator.mat', 'system')
load('system_repressilator.mat')

                                                                            % setup generator
generator = Generator(system, N=1, t=0:5:100, error_std=.05, D_mult=eps, observed=1:3);

nseeds = 100;                                                               % number of repetitions
[estimates_EGM, estimates_TM] = deal(zeros(nseeds, system.P));              % allocate parameter estimates, ICs,
[ICs_EGM, ICs_TM] = deal(zeros(nseeds, system.K));                          % computation times, and model fits
[times_EGM, times_TM] = deal(zeros(nseeds, 2));
[fits_EGM, fits_TM] = deal(zeros(nseeds, 1));

for seed = 1:nseeds
    rng(seed)
    [data, ground_truth] = generator.generate();                            % generate data
    
                                                                            % setup estimator
    estimator = EGM(system, data, Knots=10:10:90, InitialConditions=[10 20 30 .9 .9 .9 .2 .2 .2]);
    
    out1 = estimator.estimate(.5*system.k0');                               % estimate using EGM
    out2 = estimator.estimate_TM(.5*system.k0', [10 20 30 .9 .9 .9 .2 .2 .2]);        % estimate using TM
    
    estimates_EGM(seed, :) = out1.beta;                                     % store parameter estimates and ICs
    estimates_TM(seed, :) = out2.beta;
    ICs_EGM(seed, :) = out1.init;
    ICs_TM(seed, :) = out2.init;

    times_EGM(seed, :) = out1.time;                                         % store computation times
    times_TM(seed, :) = out2.time;

    discrepancies_EGM = (data.traces - out1.fitted(:, data.observed));      % store model fits
    discrepancies_TM = (data.traces - out2.fitted(:, data.observed));
    fits_EGM(seed) = mean(discrepancies_EGM.^2 ./ ground_truth.original(:, data.observed), 'all');
    fits_TM(seed) = mean(discrepancies_TM.^2 ./ ground_truth.original(:, data.observed), 'all');
end

save('simulation/test_repressilator.mat')                                   % store results
