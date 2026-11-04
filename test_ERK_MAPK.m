%% Setup system ------------------------------------------------------------
clearvars
% Microscopic (mass-action) Markevich et al. (2004) MAPK model with BioModels
% parameter values, run in the QSSA regime: MAPKK(0) = 1 (was 50), MKP3(0) = 2
% (was 100), M(0) = 150 (was 500). Dynamics are ~50x slower than with the
% original enzyme levels, hence the 50x longer time grid (0:250:5000 s).
%
% Measurements: y1 = Mpt  = Mp + Mp-KK + Mp-P + Mp-P*  (state 2)
%               y2 = Mppt = Mpp + Mpp-P                 (state 3)
%
% Parameters equal to 1.0 (km1, km3, hm1, h3, hm4) set the relative time
% scales and are always fixed. With three experiments (kinase-only,
% phosphatase-only, both), fixing k4, h1, h5 and hm3 as well leaves
% k1, k2, k3, h2, h4, h6, hm6 free. EGM handles a single experiment only, so
% by default more parameters are fixed: a local (FIM-based) analysis of this
% single experiment (5% multiplicative noise, t = 0:250:5000) gives relative
% SDs of up to ~1e2 for the 7-parameter set, ~0.9 with 5 free parameters, and
% <0.1 for the 4 free parameters k2, k3, h2, h4 used here.

multi_experiment = false;                                                   % true: 7 free parameters (needs 3 experiments)

fixed_timescales = ["km1" "km3" "hm1" "h3" "hm4"];                          % parameters equal to 1.0
if multi_experiment
    fixed = [fixed_timescales "k4" "h1" "h5" "hm3"];                        % free: k1 k2 k3 h2 h4 h6 hm6
else
    fixed = [fixed_timescales "k4" "h1" "h5" "hm3" "k1" "h6" "hm6"];        % free: k2 k3 h2 h4
end
                                                                            % generate ODE system object
% system = System('model_ERK_MAPK.txt', 'auxiliary_ERK_MAPK.txt', FixedParameters=fixed);
% save('system_ERK_MAPK.mat', 'system')
load('system_ERK_MAPK.mat')

                                                                            % setup generator
generator = Generator(system, N=1, t=0:250:5000, error_std=.05, D_mult=eps, observed=["Mpt" "Mppt"]);

nseeds = 200;                                                              % number of repetitions
[estimates_EGM, estimates_TM, beta0_EGM] = deal(zeros(nseeds, system.P));   % allocate parameter estimates, ICs,
[ICs_EGM, ICs_TM] = deal(zeros(nseeds, system.K));                          % computation times, and model fits
[times_EGM, times_TM] = deal(zeros(nseeds, 2));
[fits_EGM, fits_TM] = deal(zeros(nseeds, 1));

error_sds = .25;

for error_idx = 1:length(error_sds)
    for seed = 1:nseeds
        rng(seed)
        [data, ground_truth] = generator.generate();                            % generate data

        % figure
        % plot(data.t, ground_truth.original(:, data.observed))

        error = exp(normrnd(0, error_sds(error_idx), 1, system.K));
        % error = [1/3 0 0 2 0.5 0 0 0 0 0 0]
        x0_init = error .* system.x0' %#ok<NOPTS>


                                                                                % setup estimator
        estimator = EGM(system, data, Knots=500:500:4500, InitialConditions=x0_init);

%   best to worst: 3 2 4 1
% 1: up/down  2: up--/down- 3: down- 4: down-

        error = exp([1 1 1 1] .* normrnd(0, error_sds(error_idx), 1, system.P));
        beta_init = error .* system.k0' %#ok<NOPTS>

        try
            disp('EGM')
            out1 = estimator.estimate(beta_init);                           % estimate using EGM

            beta0_EGM(seed, :, error_idx) = beta_init;
            estimates_EGM(seed, :, error_idx) = out1.beta;                  % store parameter estimates and ICs
            ICs_EGM(seed, :, error_idx) = out1.init;
            times_EGM(seed, :, error_idx) = out1.time;                      % store computation times
            discrepancies_EGM = (data.traces - out1.fitted(:, data.observed));% store model fits
            fits_EGM(seed, error_idx) = mean(discrepancies_EGM.^2 ./ ...
                                                ground_truth.original(:, data.observed), 'all');
        catch ME
            disp(ME)                                                        % catch any errors, record failure
            estimates_EGM(seed, :, error_idx) = 99*ones(size(system.k0'));
            times_EGM(seed, :, error_idx) = [99 99];
            fits_EGM(seed, error_idx) = 99;
            ICs_EGM(seed, :, error_idx) = 99*ones(size(system.x0'));
        end

        try
            disp('TM')
            out2 = estimator.estimate_TM(beta_init, x0_init + 1e-4);  % estimate using TM

            estimates_TM(seed, :, error_idx) = out2.beta;
            ICs_TM(seed, :, error_idx) = out2.init;
            times_TM(seed, :, error_idx) = out2.time;
            discrepancies_TM = (data.traces - out2.fitted(:, data.observed));
            fits_TM(seed, error_idx) = mean(discrepancies_TM.^2 ./ ...
                                                ground_truth.original(:, data.observed), 'all');
        catch ME
            % disp(ME)                                                        % catch any errors, record failure
            estimates_TM(seed, :, error_idx) = 99*ones(size(system.k0'));
            times_TM(seed, :, error_idx) = [99 99];
            fits_TM(seed, error_idx) = 99;
            ICs_TM(seed, :, error_idx) = 99*ones(size(system.x0'));
        end
    
        % try
        %     disp('GP')
        %     out3 = estimator.estimate_GP(beta_init, [10 20 30 .9 .9 .9 .2 .2 .2]);  % estimate using TM
        % 
        %     estimates_GP(seed, :, error_idx) = out3.beta;
        %     ICs_GP(seed, :, error_idx) = out3.init;
        %     times_GP(seed, :, error_idx) = out3.time;
        %     discrepancies_GP = (data.traces - out3.fitted(:, data.observed));
        %     fits_GP(seed, error_idx) = mean(discrepancies_TM.^2 ./ ...
        %                                         ground_truth.original(:, data.observed), 'all');
        % catch ME
        %     % disp(ME)                                                        % catch any errors, record failure
        %     estimates_GP(seed, :, error_idx) = 99*ones(size(system.k0'));
        %     times_GP(seed, :, error_idx) = [99 99];
        %     fits_GP(seed, error_idx) = 99;
        %     ICs_GP(seed, :, error_idx) = 99*ones(size(system.x0'));
        % end
    end
end

save('simulation/test_ERK_MAPK_2.mat')                                        % store results


%%
close all

for rep = 1:nseeds, accuracies_EGM(rep) = eucl_rel(system.k0', estimates_EGM(rep, :)); end
for rep = 1:nseeds, accuracies_TM(rep) = eucl_rel(system.k0', estimates_TM(rep, :)); end
summary_EGM = quantile(accuracies_EGM, [.05 .5 .95])
summary_TM = quantile(accuracies_TM, [.05 .5 .95])

figure
title('Initialization/Accuracy')
for i = 1:4
    for j = 1:4
        p = subplot(4, 4, (i-1)*4 + j);
        if i ~= j
            scatter(beta0_EGM(:, i), beta0_EGM(:, j), [], accuracies_EGM); % Color by 'colorValues'
        else
            p.Title.String = system.parameters_variable(i);
            histogram(beta0_EGM(:, i))
        end
    end
end

figure
title('Initialization/Fit')
for i = 1:4
    for j = 1:4
        p = subplot(4, 4, (i-1)*4 + j);
        if i ~= j
            scatter(beta0_EGM(:, i), beta0_EGM(:, j), [], min(3, fits_EGM)); % Color by 'colorValues'
        else
            p.Title.String = system.parameters_variable(i);
            histogram(beta0_EGM(:, i))
        end
    end
end

figure
title('Final/Fit')
for i = 1:4
    for j = 1:4
        p = subplot(4, 4, (i-1)*4 + j);
        if i ~= j
            scatter(estimates_EGM(:, i), estimates_EGM(:, j), [], min(3, fits_EGM)); % Color by 'colorValues'
        else
            p.Title.String = system.parameters_variable(i);
            histogram(estimates_EGM(:, i))
        end
    end
end