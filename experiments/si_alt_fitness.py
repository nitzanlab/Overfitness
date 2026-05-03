"""
SI Fig S10, S11: selected complexity in stable / changing environments under
alternative fitness functions (L1, rational, cutoff).

Outputs:
- data/s10_alt_fitness.npz       : bubble matrices for static envs.
- data/s11_alt_fitness_change.npz: heatmaps over (q*, env_change_rate).

Source: sensitivity_analysis.ipynb §1.1.
"""
import numpy as np
from overfitness_paper import (
    run, np_temp_seed, set_fitness_type, run_figure_experiment,
)
from overfitness_paper.config import (
    FITNESS_GAMMA, SIGMA, T, CLASS_SIZE, N, KS, N_REALIZATIONS,
)
from ._paths import data_path

FITNESS_TYPES = ['L2', 'L1', 'rational', 'cutoff']


def static_bubble(fitness_type, n_realizations):
    set_fitness_type(fitness_type)
    n_ks = len(KS)
    bubble = np.zeros((n_ks, n_ks))
    for i, q_star in enumerate(KS):
        for r in range(n_realizations):
            with np_temp_seed(100000 + i * n_realizations + r):
                log = run(
                    exp_name=f's10_{fitness_type}_q{q_star}_r{r}',
                    true_k=q_star, fitness_gamma=FITNESS_GAMMA,
                    n=N, T=T, class_size=CLASS_SIZE, xi=SIGMA,
                    ks=KS, to_return=['class_frequency'],
                )
            sel = KS[int(np.argmax(log['class_frequency'].mean(axis=0)))]
            bubble[i, KS.index(sel)] += 1.0 / n_realizations
    return bubble


def main():
    bubbles = {}
    for ft in FITNESS_TYPES:
        print(f'S10 fitness={ft}')
        bubbles[ft] = static_bubble(ft, N_REALIZATIONS)
    set_fitness_type('L2')
    np.savez(data_path('s10_alt_fitness.npz'), ks=np.array(KS), **bubbles)
    print('Saved data/s10_alt_fitness.npz')

    n_envs_list = [1, 5, 10, 20, 30, 40]
    heatmaps = {}
    for ft in FITNESS_TYPES:
        print(f'S11 fitness={ft}')
        set_fitness_type(ft)
        all_freqs = run_figure_experiment(
            true_ks=KS, n_envs_list=n_envs_list,
            n_realizations=N_REALIZATIONS, T=T,
            class_size=CLASS_SIZE, fitness_gamma=FITNESS_GAMMA,
            xi=SIGMA, function_type='linear',
        )
        sel_idx = np.argmax(all_freqs.mean(axis=3), axis=-1)
        heatmaps[ft] = np.array(KS)[sel_idx].mean(axis=-1)
    set_fitness_type('L2')
    np.savez(data_path('s11_alt_fitness_change.npz'),
             ks=np.array(KS),
             n_envs_list=np.array(n_envs_list),
             **heatmaps)
    print('Saved data/s11_alt_fitness_change.npz')


if __name__ == '__main__':
    main()
