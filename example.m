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

figure                                                                      % plot generated data
cols = num2cell(cool(3), 2);
h = plot(data.t, ground_truth.original, ':', LineWidth=1.5);
set(h, {'color'}, cols);
hold on
plot(data.t, data.traces, 'o-', color=cols{3})
legend('x_1', 'x_2', 'x_3', 'y')
xlabel('Time (min)')
ylabel('Concentration (pM)')
title('Generated dynamics and measurements')


%% Infer using EGM ---------------------------------------------------------
ICmean0 = (1+error)*system.x0';                                             % initial IC mean guess
estimator = EGM(system, data, Knots=2.5:1.25:7.5, ICmean=ICmean0);

beta0 = [0.5 1.0];                                                          % initial model parameter guess
out = estimator.estimate(beta0);

beta_est = out.beta                                                         % final parameter estimate

h2 = plot(out.t_fine, out.fitted_fine, '--');                               % plot fitted 
set(h2, {'color'}, cols);   
legend('x_1', 'x_2', 'x_3', 'y', 'x̂_1', 'x̂_2', 'x̂_3')