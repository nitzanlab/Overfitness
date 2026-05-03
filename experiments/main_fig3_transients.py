"""
Main Fig 3: Transient success of simple classes.

Outputs:
- data/fig3_dynamics.npz       : per-timestep, per-class metrics averaged over
                                 N_REALIZATIONS realizations at q*=5 (linear).
- data/fig3_fitness_dist.npz   : per-class fitness distributions for the 3D plot
                                 in panel B (sampled at one representative timestep).

Source: derived from cells in `reproduce_figures.ipynb` §3.
"""
import numpy as np
from overfitness_paper import run_with_metrics, np_temp_seed
from overfitness_paper.config import (
    FITNESS_GAMMA, SIGMA, T, CLASS_SIZE, N, KS, N_REALIZATIONS,
)
from ._paths import data_path

T_FIG3 = 500
Q_STAR = 5


def main():
    n_ks = len(KS)
    class_freqs = np.zeros((N_REALIZATIONS, T_FIG3, n_ks))
    class_fitness = np.zeros((N_REALIZATIONS, T_FIG3, n_ks))
    max_fit = np.zeros((N_REALIZATIONS, T_FIG3, n_ks))
    occam = np.zeros((N_REALIZATIONS, T_FIG3, n_ks))
    growth = np.zeros((N_REALIZATIONS, T_FIG3, n_ks))

    for r in range(N_REALIZATIONS):
        with np_temp_seed(2000 + r):
            log = run_with_metrics(
                exp_name=f'fig3_r{r}',
                true_k=Q_STAR, fitness_gamma=FITNESS_GAMMA,
                n=N, T=T_FIG3, class_size=CLASS_SIZE, xi=SIGMA,
                ks=KS,
            )
        class_freqs[r] = log['class_frequency']
        class_fitness[r] = log['class_fitness']
        max_fit[r] = log['max_fitness_per_class']
        occam[r] = log['occam_factor']
        growth[r] = log['class_growth_rate']

    np.savez(data_path('fig3_dynamics.npz'),
             ks=np.array(KS), q_star=Q_STAR,
             class_frequency=class_freqs,
             class_fitness=class_fitness,
             max_fitness=max_fit,
             occam_factor=occam,
             class_growth_rate=growth)
    print('Saved data/fig3_dynamics.npz')

    # For Fig 3B: per-class fitness summary at t=0 (uniform composition).
    # The 3D surface is built from per-class mean/std curves; the original
    # notebook (`reproduce_figures.ipynb` §3) builds it directly from the
    # per-class fitness time series we already have above.
    np.savez(data_path('fig3_fitness_dist.npz'),
             ks=np.array(KS),
             class_fitness_t0=class_fitness[:, 0, :])  # (n_real, n_ks)
    print('Saved data/fig3_fitness_dist.npz')


if __name__ == '__main__':
    main()
