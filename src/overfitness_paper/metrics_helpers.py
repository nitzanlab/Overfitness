"""
Convenience wrapper that runs a simulation and computes per-timestep
metrics (class fitness, max fitness per class, Occam factor, class growth rate)
that are not directly logged by `simulation.run()`.

This avoids reimplementing the same loop in every experiment script.
"""

import numpy as np
from .simulation import (
    run,
    compute_class_fitness, compute_max_fitness_per_class,
    compute_occam_factor, compute_class_growth_rate,
)


def run_with_metrics(*, ks, class_size, n_envs=None, env_change_rate=None,
                     **run_kwargs):
    """
    Run a simulation and compute per-timestep metrics.

    Returns
    -------
    dict with keys:
        'class_frequency'        (T, n_ks)
        'class_fitness'          (T, n_ks)
        'max_fitness_per_class'  (T, n_ks)
        'occam_factor'           (T-1, n_ks)  # only T-1 because needs t+1
        'class_growth_rate'      (T-1, n_ks)
        'optimal_member_fitness' (T, n_ks)    # alias of max_fitness_per_class

    Note: Occam factor and class growth rate use the same length (T-1) so
    we pad them with NaN at the last timestep to match (T, n_ks) shapes
    for consistency with the simulation outputs.
    """
    raw = run(
        ks=ks, class_size=class_size,
        n_envs=n_envs, env_change_rate=env_change_rate,
        to_return=['frequency', 'class_frequency', 'fitness'],
        **run_kwargs,
    )
    freq = raw['frequency']        # (T, n_types)
    class_freq = raw['class_frequency']  # (T, n_ks)
    fitness = raw['fitness']       # (T, n_types)
    T = freq.shape[0]
    n_ks = len(ks)
    class_size_list = (
        class_size if isinstance(class_size, list) else [class_size] * n_ks
    )

    class_fit = np.zeros((T, n_ks))
    max_fit = np.zeros((T, n_ks))
    occam = np.full((T, n_ks), np.nan)
    growth = np.full((T, n_ks), np.nan)

    for t in range(T):
        class_fit[t] = compute_class_fitness(freq[t], fitness[t], class_size_list)
        max_fit[t] = compute_max_fitness_per_class(fitness[t], class_size_list)
        if t < T - 1:
            occam[t] = compute_occam_factor(freq[t], freq[t + 1], fitness[t],
                                            class_size_list)
            growth[t] = compute_class_growth_rate(class_freq[t + 1], class_freq[t])

    return {
        'class_frequency': class_freq,
        'class_fitness': class_fit,
        'max_fitness_per_class': max_fit,
        'optimal_member_fitness_per_class': max_fit,
        'occam_factor': occam,
        'class_growth_rate': growth,
    }
