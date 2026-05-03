"""
SI Fig S5: AR(1) correlated environments.

Three conditions × two function classes (linear, NN):
 - Static env, iid cues  (rho_env=0, change_rate=0)
 - Changing iid envs     (rho_env=0, change_rate=0.05)
 - Changing AR(1) envs   (rho_env=0.99, change_rate=0.05)

Outputs: data/s5_ar1_linear.npz, data/s5_ar1_nn.npz

Source: ar1_environments.ipynb + run_linear_correlated.py + run_nn_correlated.py.
The linear path uses the same _init_population/run_single helpers as the legacy
script; we re-implement them here in compact form using simulation.run().
"""
import numpy as np
from overfitness_paper import run, np_temp_seed
from overfitness_paper.models.neural_net import run_experiment as run_nn_experiment
from overfitness_paper.models.neural_net import compute_bubble_matrix
from overfitness_paper.config import (
    FITNESS_GAMMA, SIGMA, T, CLASS_SIZE, N, KS, N_REALIZATIONS, NN_KS,
)
from ._paths import data_path


def linear_bubble(change_rate, ar1_rho, n_realizations):
    n_ks = len(KS)
    bubble = np.zeros((n_ks, n_ks))
    for i, q_star in enumerate(KS):
        for r in range(n_realizations):
            with np_temp_seed(50000 + i * n_realizations + r):
                log = run(
                    exp_name=f's5_q{q_star}_cr{change_rate}_rho{ar1_rho}_r{r}',
                    true_k=q_star, fitness_gamma=FITNESS_GAMMA,
                    n=N, T=T, class_size=CLASS_SIZE, xi=SIGMA,
                    ks=KS, to_return=['class_frequency'],
                    env_change_rate=change_rate, ar1_rho=ar1_rho,
                )
            sel = KS[int(np.argmax(log['class_frequency'].mean(axis=0)))]
            bubble[i, KS.index(sel)] += 1.0 / n_realizations
    return bubble


def main():
    n_real = max(N_REALIZATIONS // 2, 50)
    bubbles = {}
    for label, change_rate, rho in [
        ('static_iid', 0.0, 0.0),
        ('changing_iid', 0.05, 0.0),
        ('changing_ar1_99', 0.05, 0.99),
    ]:
        print(f'Linear: {label}')
        bubbles[label] = linear_bubble(change_rate, rho, n_real)
    np.savez(data_path('s5_ar1_linear.npz'),
             ks=np.array(KS), **bubbles)
    print('Saved data/s5_ar1_linear.npz')

    nn_bubbles = {}
    for label, change_rate, rho in [
        ('static_iid', 0.0, 0.0),
        ('changing_iid', 0.05, 0.0),
        ('changing_ar1_99', 0.05, 0.99),
    ]:
        print(f'NN: {label}')
        # NB: the NN runner does not yet support env AR(1); ar1>0 falls back to
        # rapidly switching uncorrelated envs. See models/neural_net.py.
        d = run_nn_experiment(
            true_ks=NN_KS, ks=NN_KS, class_size=CLASS_SIZE,
            T=T, fitness_gamma=0.01, xi=1.0,
            n_in=N, n_out=N,
            change_rate=change_rate, cue_rho=0,
            n_realizations=n_real,
        )
        nn_bubbles[label] = compute_bubble_matrix(d, NN_KS)
    np.savez(data_path('s5_ar1_nn.npz'),
             ks=np.array(NN_KS), **nn_bubbles)
    print('Saved data/s5_ar1_nn.npz')


if __name__ == '__main__':
    main()
