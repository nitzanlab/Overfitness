"""
SI Fig S3: polynomial function class — selected complexity in stable and
changing environments.

Outputs:
- data/s3_polynomial.npz : bubble matrix (S3A) + heatmap of selected complexity
                           vs (q*, env_change_rate) (S3B).

Source: `evo_complexity_polynomial.ipynb`. The original notebook used
`bayesian_evolution.py`; this script uses the unified `simulation.run` engine
with `set_function_type('polynomial')`.
"""
import numpy as np
from overfitness_paper import (
    run, np_temp_seed, set_function_type,
    run_figure_experiment,
)
from overfitness_paper.config import FITNESS_GAMMA, SIGMA, T, KS, N_REALIZATIONS
from ._paths import data_path

CLASS_SIZE_POLY = 500
SIGMA_POLY = 0.1


def main():
    set_function_type('polynomial')

    # S3A: bubble of selected complexity in static envs
    n_ks = len(KS)
    bubble = np.zeros((n_ks, n_ks))
    for i, q_star in enumerate(KS):
        for r in range(N_REALIZATIONS):
            with np_temp_seed(40000 + i * N_REALIZATIONS + r):
                log = run(
                    exp_name=f's3_q{q_star}_r{r}',
                    true_k=q_star, fitness_gamma=FITNESS_GAMMA,
                    n=3, T=T, class_size=CLASS_SIZE_POLY, xi=SIGMA_POLY,
                    ks=KS, to_return=['class_frequency'],
                )
            sel = KS[int(np.argmax(log['class_frequency'].mean(axis=0)))]
            bubble[i, KS.index(sel)] += 1.0 / N_REALIZATIONS

    # S3B: heatmap
    n_envs_list = [1, 5, 10, 25, 50, 100, 200, 500, 1000]
    all_freqs = run_figure_experiment(
        true_ks=KS, n_envs_list=n_envs_list,
        n_realizations=N_REALIZATIONS, T=T, class_size=CLASS_SIZE_POLY,
        fitness_gamma=FITNESS_GAMMA, xi=SIGMA_POLY, function_type='polynomial',
    )
    avg_over_t = all_freqs.mean(axis=3)
    sel_idx = np.argmax(avg_over_t, axis=-1)
    ks_arr = np.array(KS)
    heatmap = ks_arr[sel_idx].mean(axis=-1)

    set_function_type('linear')  # restore default
    np.savez(data_path('s3_polynomial.npz'),
             ks=np.array(KS), bubble=bubble,
             n_envs_list=np.array(n_envs_list), heatmap=heatmap)
    print('Saved data/s3_polynomial.npz')


if __name__ == '__main__':
    main()
