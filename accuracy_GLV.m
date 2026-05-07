%% Setup system ------------------------------------------------------------
clearvars
                                                                            % generate ODE system object
% system = System('model_GLV.txt', 'auxiliary_GLV.txt', FixedParameters = ["r"]);
% save('system_GLV.mat', 'system')
load('system_GLV.mat')

dt_values = [.25 .5 1];                                                     % experimental parameters
noise_levels = [.005 .02 .05 .1];
init_error = 5.^(-1:.5:1) - 1;                                              % IC (mean) initialization errors
seeds = 1:50;

beta0 = .5*system.k0';                                                      % initial parameter estimate

                                                                            % allocate
[betas_EGM, times_EGM, betas_GM, times_GM, betas_TM, times_TM] = deal(zeros(2, length(seeds), length(init_error), ...
                                                                            length(noise_levels), length(dt_values)));
[accuracies_EGM, accuracies_GM, accuracies_TM, sigmas, PCVs] = deal(zeros(length(seeds), length(init_error), ...
                                                                    length(noise_levels), length(dt_values)));
%% Simulate and estimate ---------------------------------------------------
date_time = string(datetime('now','Format','MM-dd_HH_mm_ss'))

for dt_idx = 1:length(dt_values)
for noise_idx = 1:length(noise_levels)
for init_idx = 1:length(init_error)
for seed = 1:length(seeds)

    dt = dt_values(dt_idx);                                                 % abbreviate
    noise = noise_levels(noise_idx);
    error = init_error(init_idx);
    dt_noise_error_seed = [dt noise error seed]

    rng(10000*dt_idx + 1000*noise_idx + 100*init_idx + seed - 1)            % set random number generator

                                                                            % generate measurements
    generator = Generator(system, N=1, t=0:dt:10, error_std=noise, D_mult=eps, observed=system.K);
    [data, ground_truth] = generator.generate();

    dt_gm = .5 + .5*(dt_idx==3);                                            % setup estimator (subsample at dt=.25)
    estimator = EGM(system, data, Knots=2:2:8, InitialConditions=(1+error)*system.x0', TimePoints=0:dt_gm:10);

    try 
        out = estimator.estimate(beta0);                                    % estimate using EGM

        betas_EGM(:, seed, init_idx, noise_idx, dt_idx) = out.beta;         % record estimate, accuracy, time
        times_EGM(:, seed, init_idx, noise_idx, dt_idx) = out.time;
        accuracies_EGM(seed, init_idx, noise_idx, dt_idx) = eucl_rel(system.k0, out.beta(:));

    catch ME    
        disp(ME)                                                            % catch any errors, record failure
        times_EGM(:, seed, init_idx, noise_idx, dt_idx) = [99 99];
        accuracies_EGM(seed, init_idx, noise_idx, dt_idx) = 99;
    end

    try 
        out2 = estimator.estimate_TM(beta0, (1+error)*system.x0');          % estimate using TM

        betas_TM(:, seed, init_idx, noise_idx, dt_idx) = out2.beta;         % record estimate, accuracy, time  
        times_TM(:, seed, init_idx, noise_idx, dt_idx) = out2.time;
        accuracies_TM(seed, init_idx, noise_idx, dt_idx) = eucl_rel(system.k0, out2.beta(:));

    catch ME
        disp(ME)                                                            % catch any errors, record failure
        times_TM(:, seed, init_idx, noise_idx, dt_idx) = [99 99];
        accuracies_TM(seed, init_idx, noise_idx, dt_idx) = 99;
    end
end
end
end
    save("simulation/accuracy_GLV.mat")                                     % save results
end
