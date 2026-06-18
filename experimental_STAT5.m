%% Setup system and load data ----------------------------------------------
clearvars
                                                                            % generate ODE system object
% system = System('model_STAT5.txt', 'auxiliary_STAT5.txt', ...
%                 FixedParameters=["tauinv" "k2"]);
% save('system_STAT5.mat', 'system')
load('system_STAT5.mat')

ty = readmatrix('STAT5-data.csv');                                          % retrieve experimental observations
t = ty(:, 1)';                                                              % measurement times
y = ty(:, 2:end);                                                           % observations
k6 = .0255;                                                                 % pre-tuned data scale (A.U.)
y(:, 2) = k6*y(:, 2);                                                       % create data object for EGM inference
data = struct('t', t(2:end), 'traces', y(2:end, :), 'init', [1 .1 .01 .001 0 0 0]+1E-8, 'observed', 1:2);

                                                                            % setup estimator
estimator = EGM(system, data, Knots={[12 18 30 40], [6 10 14 18 25 40]}, ...    % different knots per state
                InitialConditions=[1 .1 .01 .001 0 0 0]+1E-8, sigma=.001);      % crude IC guess

beta0 = .5*system.k0';                                                      % initialize parameter estimate
out1 = estimator.estimate(beta0);                                           % estimate using EGM
k1234_EGM = out1.beta;

out2 = estimator.estimate_TM(beta0, [1 .1 .01 .001 0 0 0]+1E-8);            % estimate using TM
k1234_TM = out2.beta;

k1234 = system.k0'                                                          % display estimates from Swameye et al. (2003)
k1234_EGM
k1234_TM

save('simulation/experimental.mat')                                         % store results
