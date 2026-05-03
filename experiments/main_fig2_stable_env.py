"""
Main Fig 2: Selected complexity in stable environments (linear maps).

Outputs:
- data/fig2_bubbles.npz       : two bubble matrices (max-fitness class, selected class)
                                across q* in 1..9, averaged over 100 realizations.
- data/fig2_dynamics_q5.npz   : time-averaged metrics for q*=5 used in panels C-F:
                                max_fitness[q], occam[q], growth_rate[q],
                                fitness_of_optimal_member[q].

Source: derived from cells in `reproduce_figures.ipynb` §2 (paper repo).
"""
import numpy as np
from overfitness_paper import np_temp_seed, run_with_metrics
from overfitness_paper.config import (
    FITNESS_GAMMA, SIGMA, T, CLASS_SIZE, N, KS, N_REALIZATIONS,
)
from ._paths import data_path


def run_q_star(true_k, n_realizations, T, class_size, fitness_gamma, sigma, n, ks):
    """Return (max_fitness_class[r], selected_class[r]) over n_realizations."""
    n_ks = len(ks)
    max_class = np.zeros(n_realizations, dtype=int)
    sel_class = np.zeros(n_realizations, dtype=int)

    for r in range(n_realizations):
        with np_temp_seed(r):
            log = run_with_metrics(
                exp_name=f'fig2_q{true_k}_r{r}',
                true_k=true_k, fitness_gamma=fitness_gamma,
                n=n, T=T, class_size=class_size, xi=sigma,
                ks=ks,
            )
        avg_class_freq = log['class_frequency'].mean(axis=0)
        sel_class[r] = ks[int(np.argmax(avg_class_freq))]
        time_avg_max = log['max_fitness_per_class'].mean(axis=0)
        max_class[r] = ks[int(np.argmax(time_avg_max))]

    return max_class, sel_class


def main():
    ks = KS
    n_ks = len(ks)
    bubble_max = np.zeros((n_ks, n_ks))   # rows = q*, cols = class with max fitness
    bubble_sel = np.zeros((n_ks, n_ks))

    for i, q_star in enumerate(ks):
        print(f'q*={q_star}')
        max_c, sel_c = run_q_star(
            true_k=q_star, n_realizations=N_REALIZATIONS, T=T,
            class_size=CLASS_SIZE, fitness_gamma=FITNESS_GAMMA,
            sigma=SIGMA, n=N, ks=ks,
        )
        for j, q in enumerate(ks):
            bubble_max[i, j] = np.mean(max_c == q)
            bubble_sel[i, j] = np.mean(sel_c == q)

    np.savez(data_path('fig2_bubbles.npz'),
             ks=np.array(ks),
             bubble_max_fitness_class=bubble_max,
             bubble_selected_class=bubble_sel)
    print('Saved data/fig2_bubbles.npz')

    # Panels C-F: q*=5 dynamics, time-averaged across 100 realizations.
    q_star = 5
    max_fit = np.zeros((N_REALIZATIONS, n_ks))
    occam = np.zeros((N_REALIZATIONS, n_ks))
    growth = np.zeros((N_REALIZATIONS, n_ks))
    opt_fit = np.zeros((N_REALIZATIONS, n_ks))

    for r in range(N_REALIZATIONS):
        with np_temp_seed(1000 + r):
            log = run_with_metrics(
                exp_name=f'fig2_dyn_r{r}',
                true_k=q_star, fitness_gamma=FITNESS_GAMMA,
                n=N, T=T, class_size=CLASS_SIZE, xi=SIGMA,
                ks=ks,
            )
        max_fit[r] = log['max_fitness_per_class'].mean(axis=0)
        occam[r] = np.nanmean(log['occam_factor'], axis=0)
        growth[r] = np.nanmean(log['class_growth_rate'], axis=0)
        opt_fit[r] = log['optimal_member_fitness_per_class'].mean(axis=0)

    np.savez(data_path('fig2_dynamics_q5.npz'),
             ks=np.array(ks),
             max_fitness=max_fit,
             occam_factor=occam,
             class_growth_rate=growth,
             optimal_member_fitness=opt_fit)
    print('Saved data/fig2_dynamics_q5.npz')


if __name__ == '__main__':
    main()
