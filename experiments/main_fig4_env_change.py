"""
Main Fig 4: Environmental change selects for reduced complexity.

Outputs:
- data/fig4_mullers.pkl    : two single-realization Muller plots (panels A, B):
                             one with one mid-simulation env change at t=50,
                             one with rapid changes every 50 timesteps.
- data/fig4_dynamics.npz   : per-class time series under rapid env change (panels C-D).
- data/fig4e_heatmap.npz   : selected complexity averaged over realizations,
                             as a function of (q*, env_change_rate). Panel E.

Source: derived from cells in `reproduce_figures.ipynb` §4. The heatmap uses
`run_figure_experiment` from the simulation module.
"""
import pickle
import numpy as np
from overfitness_paper import run, np_temp_seed, run_figure_experiment, run_with_metrics
from overfitness_paper.config import (
    FITNESS_GAMMA, SIGMA, T, CLASS_SIZE, N, KS, N_REALIZATIONS,
)
from ._paths import data_path


def _muller(seed, true_k, T_run, n_envs):
    with np_temp_seed(seed):
        log = run(
            exp_name=f'fig4_muller_seed{seed}_envs{n_envs}',
            true_k=true_k, fitness_gamma=FITNESS_GAMMA,
            n=N, T=T_run, class_size=CLASS_SIZE, xi=SIGMA,
            ks=KS, n_envs=n_envs,
            to_return=['class_frequency'],
        )
    return log


def main():
    # Panels A, B: single-realization Mullers
    log_single = _muller(seed=1, true_k=5, T_run=200, n_envs=2)
    log_rapid = _muller(seed=1, true_k=5, T_run=200, n_envs=4)
    with open(data_path('fig4_mullers.pkl'), 'wb') as f:
        pickle.dump({'single_change': log_single, 'rapid_change': log_rapid}, f)
    print('Saved data/fig4_mullers.pkl')

    # Panels C, D: per-timestep dynamics under rapid env change at q*=5.
    n_ks = len(KS)
    T_dyn = 500
    n_envs = T_dyn // 50
    metrics = {
        'class_frequency': np.zeros((N_REALIZATIONS, T_dyn, n_ks)),
        'class_fitness': np.zeros((N_REALIZATIONS, T_dyn, n_ks)),
        'max_fitness_per_class': np.zeros((N_REALIZATIONS, T_dyn, n_ks)),
        'occam_factor': np.zeros((N_REALIZATIONS, T_dyn, n_ks)),
        'class_growth_rate': np.zeros((N_REALIZATIONS, T_dyn, n_ks)),
    }
    for r in range(N_REALIZATIONS):
        with np_temp_seed(3000 + r):
            log = run_with_metrics(
                exp_name=f'fig4_dyn_r{r}',
                true_k=5, fitness_gamma=FITNESS_GAMMA,
                n=N, T=T_dyn, class_size=CLASS_SIZE, xi=SIGMA,
                ks=KS, n_envs=n_envs,
            )
        for k in metrics:
            metrics[k][r] = log[k]
    np.savez(data_path('fig4_dynamics.npz'), ks=np.array(KS), n_envs=n_envs, **metrics)
    print('Saved data/fig4_dynamics.npz')

    # Panel E: heatmap of selected complexity.
    n_envs_list = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    all_freqs = run_figure_experiment(
        true_ks=KS, n_envs_list=n_envs_list,
        n_realizations=N_REALIZATIONS, T=T, class_size=CLASS_SIZE,
        fitness_gamma=FITNESS_GAMMA, xi=SIGMA, function_type='linear',
    )
    # all_freqs shape: (len(n_envs_list), len(true_ks), n_realizations, T, len(ks))
    # Selected complexity = class with highest time-averaged frequency.
    avg_over_t = all_freqs.mean(axis=3)                # (n_envs, n_qstar, n_real, n_ks)
    sel_idx = np.argmax(avg_over_t, axis=-1)            # (n_envs, n_qstar, n_real)
    ks_arr = np.array(KS)
    mean_selected = ks_arr[sel_idx].mean(axis=-1)       # (n_envs, n_qstar)
    np.savez(data_path('fig4e_heatmap.npz'),
             ks=np.array(KS), n_envs_list=np.array(n_envs_list),
             mean_selected_class=mean_selected)
    print('Saved data/fig4e_heatmap.npz')


if __name__ == '__main__':
    main()
