%% Setup system ------------------------------------------------------------
clearvars
                                                                            % process model file
system = System('model_GLV.txt', 'auxiliary_GLV.txt', FixedParameters = ["r"]);

dt = .5;                                                                    % set experimental factors
noise = .02;
error = 1.25;

rng(0)
generator = Generator(system, N=1, t=0:dt:10, error_std=noise, ...          % generate data
                      error_const=min(.01, noise), D_mult=eps, observed=system.K);
[data, ground_truth] = generator.generate();


%% Compare the three estimators --------------------------------------------
ICmean0 = (1+error)*system.x0';                                             % initial IC mean guess
estimator = EGM(system, data, Knots=2.5:1.25:7.5, InitialConditions=ICmean0);

beta0 = [0.5 1.0];                                                          % initial model parameter guess

out_EGM = estimator.estimate(beta0);                                        % extended gradient matching
out_TM  = estimator.estimate_TM(beta0, ICmean0);                            % trajectory matching
out_GP  = estimator.estimate_GP(beta0, ICmean0);                            % generalized profiling

fprintf('\n%-22s %s\n', 'method', 'parameter estimate')
fprintf('%-22s %s\n', 'EGM', mat2str(round(out_EGM.beta, 4)))
fprintf('%-22s %s\n', 'trajectory matching', mat2str(round(out_TM.beta, 4)))
fprintf('%-22s %s\n', 'generalized profiling', mat2str(round(out_GP.beta, 4)))


%% Plot --------------------------------------------------------------------
figure
cols = num2cell(cool(3), 2);
h = plot(data.t, ground_truth.original, ':', LineWidth=1.5);
set(h, {'color'}, cols);
hold on
plot(data.t, data.traces, 'o', color=cols{3})
h2 = plot(out_GP.t_fine, out_GP.fitted_fine, '--');                         % model fit from the GP estimate
set(h2, {'color'}, cols);
xlabel('Time (min)')
ylabel('Concentration (pM)')
title('Generalized profiling')

figure                                                                      % the profiled splines and the ODE discrepancy
subplot(1, 2, 1)
plot(out_GP.t_fine, out_GP.smoothed_fine, LineWidth=1.5)
xlabel('Time (min)'), ylabel('Profiled states f_k(t)')
subplot(1, 2, 2)
plot(out_GP.t_ode, out_GP.discrepancy, LineWidth=1.5)
yline(0, 'k:')
xlabel('Time (min)'), ylabel('\xi_k(t; \beta)')
title('ODE discrepancy after profiling')
