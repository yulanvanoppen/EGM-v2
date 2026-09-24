function SS = GSS(par, data, system, estimator, filtered)
    t = data.t;

    smoothed_filtered = estimator.gm.smoothed_filtered;
    dsmoothed_dfiltered = estimator.gm.dsmoothed_dfiltered;

    G_smoothed_filtered = system.g(smoothed_filtered(2:end-1, :), t(2:end-1));
    H_smoothed_filtered = system.h(smoothed_filtered(2:end-1, :), t(2:end-1));

    design = G_smoothed_filtered;
    const = H_smoothed_filtered;
    response = dsmoothed_dfiltered(2:end-1, :) - const;

    V = estimator.gm.covariances(par, filtered);
    precision = tryinv(nearestSPD(V));
    Y = flatten(response);
    X = design;

    SS = (Y - X*par')' * precision * (Y - X*par');
end