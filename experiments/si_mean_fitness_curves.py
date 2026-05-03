"""
SI Fig S1, S2: Mean class fitness <p_q> as a function of class complexity.

S1 covers the linear function class plus two multi-peak variants (J=3, J=5).
S2 covers two-layer NNs.

For each (q*, q), we sample n_realizations classes of size class_size, evaluate
fitness against a fixed ground truth over n_signals random inputs, and average.

INCONSISTENCY (flagged, NOT fixed): the SI captions state sigma=0.1 but the
notebook code uses sigma=1.0. Per user direction we keep sigma=1.0.
See data/README.md.

Source: derived from `mean_fitness_vs_complexity.ipynb`.
"""
import numpy as np
from overfitness_paper import (
    np_temp_seed,
    sample_parameters_linear, fitness_population_linear,
    sample_environments_linear,
)
from overfitness_paper.config import FITNESS_GAMMA, SIGMA, N, KS
from ._paths import data_path

N_SIGNALS = 200
N_GROUND_TRUTH_SAMPLES = 200
CLASS_SIZE = 2000
Q_STARS = [1, 3, 5, 7, 9]


def mean_p_q_linear(q_star, q, n_signals, n_gt_samples, class_size,
                    gamma, sigma, n):
    """Mean fitness of a class-q population against a fixed q*-complexity GT."""
    p_q_samples = np.zeros(n_gt_samples)
    for s in range(n_gt_samples):
        with np_temp_seed(s):
            A_star = sample_parameters_linear(n, q_star)
            E = sample_environments_linear(n_signals, A_star, n, xi=sigma)
            pop = np.stack([sample_parameters_linear(n, q) for _ in range(class_size)])
            fits = np.zeros(n_signals)
            for t in range(n_signals):
                fits[t] = fitness_population_linear(pop, E[t], gamma=gamma).mean()
            p_q_samples[s] = fits.mean()
    return p_q_samples.mean(), p_q_samples.std() / np.sqrt(n_gt_samples)


def main():
    n_q = len(KS)
    means = np.zeros((len(Q_STARS), n_q))
    sems = np.zeros((len(Q_STARS), n_q))
    for i, q_star in enumerate(Q_STARS):
        for j, q in enumerate(KS):
            m, sem = mean_p_q_linear(
                q_star, q, N_SIGNALS, N_GROUND_TRUTH_SAMPLES,
                CLASS_SIZE, FITNESS_GAMMA, SIGMA, N,
            )
            means[i, j] = m
            sems[i, j] = sem
            print(f'  q*={q_star}, q={q}: <p_q>={m:.4f}')
    np.savez(data_path('s1_mean_fitness.npz'),
             ks=np.array(KS), q_stars=np.array(Q_STARS),
             means=means, sems=sems)
    print('Saved data/s1_mean_fitness.npz')

    # S2 (NN) is computed analogously but uses overfitness_paper.models.neural_net
    # via the run_experiment helper. See data/README.md for parameters.
    print('NOTE: S2 (NN) computed by si_neural_network.py — see that script.')


if __name__ == '__main__':
    main()
