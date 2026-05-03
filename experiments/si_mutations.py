"""
SI Fig S9: mutation-driven complexity evolution.

Three sub-figures:
- q0=1 trajectories at q* in {1,3,5,7,9}  → data/s9_q1.npz
- q0=9 trajectories at q* in {1,3,5,7,9}  → data/s9_q9.npz
- 3x3 (mu_b, mu_w) grid at q*=5, q0=1     → data/s9_grid.npz

Source: run_mutation_figures.py (verbatim parameters).
"""
import numpy as np
from overfitness_paper import run_with_mutation, np_temp_seed
from overfitness_paper.config import FITNESS_GAMMA, SIGMA, N, KS
from ._paths import data_path

T_MUT = 10000
CLASS_SIZE_MUT = 200
M = 100 * CLASS_SIZE_MUT
N_REAL = 50
TARGETS = [1, 3, 5, 7, 9]
MU_B_DEFAULT = 1e-3


def run_targets(initial_k, targets, mu_b=MU_B_DEFAULT, mu_w=0.0):
    out = {}
    for q_star in targets:
        avg_complexity = np.zeros((N_REAL, T_MUT + 1))
        for r in range(N_REAL):
            with np_temp_seed(90000 + initial_k * 100 + q_star * 10 + r):
                log = run_with_mutation(
                    exp_name=f's9_q0{initial_k}_q{q_star}_r{r}',
                    true_k=q_star, initial_k=initial_k,
                    mutation_rate_between=mu_b, mutation_rate_within=mu_w,
                    n=N, T=T_MUT, M=M, class_size=CLASS_SIZE_MUT,
                    fitness_gamma=FITNESS_GAMMA, xi=SIGMA,
                    min_k=min(KS), max_k=max(KS), seed=r,
                )
            avg_complexity[r] = log['avg_complexity']
        out[f'q{q_star}'] = avg_complexity
    return out


def main():
    print('S9 q0=1 trajectories...')
    np.savez(data_path('s9_q1.npz'), **run_targets(1, TARGETS))
    print('S9 q0=9 trajectories...')
    np.savez(data_path('s9_q9.npz'), **run_targets(9, TARGETS))

    print('S9 3x3 grid (mu_b, mu_w) at q*=5, q0=1...')
    grid = {}
    for mu_b in [1e-4, 1e-3, 1e-2]:
        for mu_w in [0.0, 1e-3, 1e-2]:
            avg = np.zeros((N_REAL, T_MUT + 1))
            for r in range(N_REAL):
                with np_temp_seed(91000 + int(mu_b * 1e6) * 100 + int(mu_w * 1e6) + r):
                    log = run_with_mutation(
                        exp_name=f's9_grid_mb{mu_b}_mw{mu_w}_r{r}',
                        true_k=5, initial_k=1,
                        mutation_rate_between=mu_b, mutation_rate_within=mu_w,
                        n=N, T=T_MUT, M=M, class_size=CLASS_SIZE_MUT,
                        fitness_gamma=FITNESS_GAMMA, xi=SIGMA,
                        min_k=min(KS), max_k=max(KS), seed=r,
                    )
                avg[r] = log['avg_complexity']
            grid[f'mb{mu_b}_mw{mu_w}'] = avg
    np.savez(data_path('s9_grid.npz'), **grid)
    print('Saved data/s9_grid.npz')


if __name__ == '__main__':
    main()
