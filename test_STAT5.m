%% Setup system ------------------------------------------------------------
clearvars
                                                                            % generate ODE system object
% system = System('model_STAT5.txt', 'auxiliary_STAT5.txt', ...
%                 FixedParameters=["tauinv"]);
% save('system_STAT5.mat', 'system')
load('system_STAT5.mat')

ty = readmatrix('STAT5-data.csv');                                          % retrieve experimental observation times
t = ty(:, 1)';
                                                                            % setup generator
generator = Generator(system, N=1, t=t, error_std=.02, D_mult=eps, observed=1:2);

nseeds = 100;                                                               % number of repetitions
[estimates_EGM, estimates_TM] = deal(zeros(nseeds, system.P));              % allocate parameter estimates, ICs,
[ICs_EGM, ICs_TM] = deal(zeros(nseeds, system.K));                          % computation times, and model fits
[times_EGM, times_TM] = deal(zeros(nseeds, 2));
[fits_EGM, fits_TM] = deal(zeros(nseeds, 1));

for seed = 1:nseeds
    rng(seed)
    [data, ground_truth] = generator.generate();                            % generate data
    data.t = data.t(2:end);                                                 % omit first data point
    data.traces = data.traces(2:end, :);
    data.T = data.T-1;
                                                                            % setup estimator
    estimator = EGM(system, data, Knots={[12 18 30 40], [6 10 14 18 25 40]}, ...    % different knots per state
                InitialConditions=[1 .1 .01 .001 0 0 0]+1E-8, sigma=.001, ...       % crude IC guess
                LB=.1*system.k0', UB=10*system.k0', ...                             % wider bounds
                Prior=struct('mean', [1 1E-8 1 1], 'prec', diag([0 4 0 0])));       % prior to prevent unbounded k2
    
    out1 = estimator.estimate(.5*system.k0');                               % estimate using EGM
    out2 = estimator.estimate_TM(.5*system.k0', [1 .1 .01 .001 0 0 0]+1E-8);% estimate using TM
    
    estimates_EGM(seed, :) = out1.beta;                                     % store parameter estimates and ICs
    estimates_TM(seed, :) = out2.beta;
    ICs_EGM(seed, :) = out1.init;
    ICs_TM(seed, :) = out2.init;

    times_EGM(seed, :) = out1.time;                                         % store computation times
    times_TM(seed, :) = out2.time;

    discrepancies_EGM = (data.traces - out1.fitted(:, data.observed));      % store model fits
    discrepancies_TM = (data.traces - out2.fitted(:, data.observed));
    fits_EGM(seed) = mean(discrepancies_EGM.^2 ./ ground_truth.original(2:end, data.observed), 'all');
    fits_TM(seed) = mean(discrepancies_TM.^2 ./ ground_truth.original(2:end, data.observed), 'all');
end

save('simulation/test_STAT5.mat')                                           % store results
