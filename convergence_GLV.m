%% Setup system ------------------------------------------------------------
clearvars
                                                                            % generate ODE system object
% system = System('model_GLV.txt', 'auxiliary_GLV.txt', FixedParameters = ["r"]);
% save('system_GLV.mat', 'system')
load('system_GLV.mat')

dt = .5;                                                                    % representative settings
noise = .05;
init_error = 5.^(-1:.5:1) - 1;                                              % IC (mean) initialization errors
seeds = 1:50;

rng(23400)                                                                  % representative sample

                                                                            % generate measurements
generator = Generator(system, N=1, t=0:dt:10, error_std=noise, error_const=.01, D_mult=eps, observed=system.K);
[data, ground_truth] = generator.generate();

magnitudes = linspace(0, 2, 21);                                            % grid of relative parameter initialization errors
converged_EGM = 999 * ones(21, 21, 4);                                      % allocate relative ground truth distances
converged_TM = 999 * ones(21, 21, 4);

for init_idx = 1:4                                                          % loop over IC errors
for b1_idx = 1:21                                                           % loop over parameter initialization grid
for b2_idx = 1:21
    error = init_error(init_idx);                                           % IC initialization error
    beta0 = [magnitudes(b1_idx) * 1, magnitudes(b2_idx) * 2];               % parameter initialization error

                                                                            % setup estimator
    estimator = EGM(system, data, Knots=2:2:8, InitialConditions=(1+error)*system.x0', TimePoints = 0:dt_gm:10);

    try 
        out = estimator.estimate(beta0);                                    % estimate using EGM
        converged_EGM(b1_idx, b2_idx, init_idx) = dist_EGM{1};              % relative distance to ground truth
        
    catch ME
        disp(ME)
    end

    try 
        out2 = estimator.estimate_TM(beta0, (1+error)*system.x0');          % estimate using EGM
        converged_TM(b1_idx, b2_idx, init_idx) = dist_TM{1};                % relative distance to ground truth
        
    catch ME
        disp(ME)
    end
end
end
end

save('simulation/convergence_GLV')                                          % save results
