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


%% Compare the estimators ----------------------------------------------------
ICmean0 = (1+error)*system.x0';                                             % initial IC mean guess
estimator = EGM(system, data, Knots=2.5:1.25:7.5, InitialConditions=ICmean0, ...
                LambdaPC=1e2, MultiStartPC=5);

beta0 = [0.5 1.0];                                                          % initial model parameter guess

out_EGM = estimator.estimate(beta0);                                        % extended gradient matching
out_TM  = estimator.estimate_TM(beta0, ICmean0);                            % trajectory matching
out_GP  = estimator.estimate_GP(beta0, ICmean0);                            % generalized profiling (Simpson tutorial)
out_PC  = estimator.estimate_PC(beta0, ICmean0);                            % parameter cascade (Ramsay original)

fprintf('\n%-26s %-22s %s\n', 'method', 'parameter estimate', 'time (s)')
fprintf('%-26s %-22s %.2f\n', 'EGM', mat2str(round(out_EGM.beta, 4)), out_EGM.time(2))
fprintf('%-26s %-22s %.2f\n', 'trajectory matching', mat2str(round(out_TM.beta, 4)), out_TM.time(2))
fprintf('%-26s %-22s %.2f\n', 'generalized profiling', mat2str(round(out_GP.beta, 4)), out_GP.time(2))
fprintf('%-26s %-22s %.2f\n', 'parameter cascade', mat2str(round(out_PC.beta, 4)), out_PC.time(2))
fprintf('%-26s %s\n', 'nominal', mat2str(round(system.k0', 4)))

fprintf('\nparameter cascade diagnostics\n')                                % the two criteria of the cascade: H is the
fprintf('  outer criterion H  = %.4e\n', out_PC.criteria_pc(1))             % data fit of the profiled states, J the
fprintf('  inner criterion J  = %.4e\n', out_PC.criteria_pc(2))             % penalized inner objective
fprintf('  lambda per equation= %s\n', mat2str(round(out_PC.lambda_pc.equation, 2)))
fprintf('  max |xi(t)|        = %.3e\n', max(abs(out_PC.discrepancy), [], 'all'))
fprintf('  estimated IC       = %s\n', mat2str(round(out_PC.init, 3)))
fprintf('  true IC            = %s\n', mat2str(round(system.x0', 3)))


%% Trajectories ---------------------------------------------------------------
cols = num2cell(cool(system.K), 2);

figure
h = plot(data.t, ground_truth.original, ':', LineWidth=1.5);                % ground truth
set(h, {'color'}, cols);
hold on
plot(data.t, data.traces, 'o', color=cols{end})                             % data
h2 = plot(out_PC.t_fine, out_PC.fitted_fine, '-', LineWidth=1.5);           % ODE solved at the cascade estimate
set(h2, {'color'}, cols);
h3 = plot(out_PC.t_fine, out_PC.smoothed_fine, '--');                       % the profiled splines themselves
set(h3, {'color'}, cols);
xlabel('Time (min)')
ylabel('Concentration (pM)')
title('Parameter cascade: solid = ODE at \theta, dashed = profiled splines')


%% Diagnostics ----------------------------------------------------------------
figure
subplot(1, 3, 1)                                                            % how far the splines are from solving the
plot(out_PC.t_ode, out_PC.discrepancy, LineWidth=1.5)                       % ODE. Large |xi| means lambda is too small
yline(0, 'k:')                                                              % (or the model is misspecified)
xlabel('Time (min)'), ylabel('\xi_k(t; \beta)')
title('ODE discrepancy')

subplot(1, 3, 2)                                                            % the outer iterations re-estimate the noise
plot(out_PC.beta_iterations, 'o-', LineWidth=1.5)                           % variances; theta should settle quickly
xlabel('Outer iteration'), ylabel('\theta')
title('Parameter path')

subplot(1, 3, 3)                                                            % GP and PC differ in the inner solve and the
bar([out_GP.beta(:) out_PC.beta(:) system.k0])                              % outer criterion, so comparing them shows
set(gca, 'XTickLabel', system.parameters_variable)                          % what the tutorial simplifications cost
legend('profiling', 'cascade', 'nominal', Location='best')
ylabel('Estimate')
title('Simpson tutorial vs Ramsay cascade')


%% Lambda sensitivity ---------------------------------------------------------
% lambda is the one hyperparameter of the cascade and is NOT estimated here:
% Ramsay et al. tune it by hand. Scan it before trusting a result; the
% estimates should be flat over a decade or two of lambda.
lambdas = [1e1 1e2 1e3 1e4];
scan = zeros(numel(lambdas), system.P);
for i = 1:numel(lambdas)
    scan_estimator = EGM(system, data, Knots=2.5:1.25:7.5, ...
                         InitialConditions=ICmean0, LambdaPC=lambdas(i), MultiStartPC=5);
    scan(i, :) = scan_estimator.estimate_PC(beta0, ICmean0).beta;
end

figure
semilogx(lambdas, scan, 'o-', LineWidth=1.5)
hold on
yline(system.k0, 'k:')
xlabel('\lambda'), ylabel('Estimate')
title('Sensitivity to the ODE penalty weight')
legend(system.parameters_variable, Location='best')
