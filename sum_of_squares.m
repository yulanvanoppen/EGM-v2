function SS = sum_of_squares(par, data, ground_truth, system)
    t = data.t;
    y = data.traces;
    obs = data.observed;
    pred = system.integrate(par, data, t);
    var = ground_truth.noisevar;

    SS = sum((y-pred(:, obs)).^2 ./ var(2:end, obs), 'all');
end