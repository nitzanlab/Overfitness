"""
SI Fig S12, S13: alternative noise distributions (uniform, laplacian).

Outputs:
- data/s12_alt_noise.npz        : bubble matrices for static envs.
- data/s13_alt_noise_change.npz : heatmaps over (q*, env_change_rate).

Source: sensitivity_analysis.ipynb §1.2.
"""
import numpy as np
from overfitness_paper import (
    run, np_temp_seed, set_noise_type, run_figure_experiment,
)
from overfitness_paper.config import (
    FITNESS_GAMMA, SIGMA, T, CLASS_SIZE, N, KS, N_REALIZATIONS,
)
from ._paths import data_path

NOISE_TYPES = ['gaussian', 'uniform', 'laplacian']


def static_bubble(noise_type, n_realizations):
    set_noise_type(noise_type)
    n_ks = len(KS)
    bubble = np.zeros((n_ks, n_ks))
    for i, q_star in enumerate(KS):
        for r in range(n_realizations):
            with np_temp_seed(110000 + i * n_realizations + r):
                log = run(
                    exp_name=f's12_{noise_type}_q{q_star}_r{r}',
                    true_k=q_star, fitness_gamma=FITNESS_GAMMA,
                    n=N, T=T, class_size=CLASS_SIZE, xi=SIGMA,
                    ks=KS, to_return=['class_frequency'],
                )
            sel = KS[int(np.argmax(log['class_frequency'].mean(axis=0)))]
            bubble[i, KS.index(sel)] += 1.0 / n_realizations
    return bubble


def main():
    bubbles = {nt: static_bubble(nt, N_REALIZATIONS) for nt in NOISE_TYPES}
    set_noise_type('gaussian')
    np.savez(data_path('s12_alt_noise.npz'), ks=np.array(KS), **bubbles)
    print('Saved data/s12_alt_noise.npz')

    n_envs_list = [1, 5, 10, 20, 30, 40]
    heatmaps = {}
    for nt in NOISE_TYPES:
        set_noise_type(nt)
        all_freqs = run_figure_experiment(
            true_ks=KS, n_envs_list=n_envs_list,
            n_realizations=N_REALIZATIONS, T=T,
            class_size=CLASS_SIZE, fitness_gamma=FITNESS_GAMMA,
            xi=SIGMA, function_type='linear',
        )
        sel_idx = np.argmax(all_freqs.mean(axis=3), axis=-1)
        heatmaps[nt] = np.array(KS)[sel_idx].mean(axis=-1)
    set_noise_type('gaussian')
    np.savez(data_path('s13_alt_noise_change.npz'),
             ks=np.array(KS),
             n_envs_list=np.array(n_envs_list),
             **heatmaps)
    print('Saved data/s13_alt_noise_change.npz')


if __name__ == '__main__':
    main()
