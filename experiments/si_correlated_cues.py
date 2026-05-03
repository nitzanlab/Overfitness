"""
SI Fig S6: temporally correlated input cues in a fixed environment.

Three conditions × two function classes (linear, NN):
 - iid cues             (rho_cue=0)
 - AR(1) correlated     (rho_cue=0.99)
 - constant cue         (rho_cue=1.0)

Outputs: data/s6_cues_linear.npz, data/s6_cues_nn.npz

Source: correlated_cues.ipynb + run_linear_correlated_cues.py + run_nn_correlated_cues.py.
"""
import numpy as np
from overfitness_paper import run, np_temp_seed
from overfitness_paper.models.neural_net import run_experiment as run_nn_experiment
from overfitness_paper.models.neural_net import compute_bubble_matrix
from overfitness_paper.config import (
    FITNESS_GAMMA, SIGMA, T, CLASS_SIZE, N, KS, N_REALIZATIONS, NN_KS,
)
from ._paths import data_path


def linear_bubble(rho_cue, n_realizations):
    n_ks = len(KS)
    bubble = np.zeros((n_ks, n_ks))
    for i, q_star in enumerate(KS):
        for r in range(n_realizations):
            with np_temp_seed(60000 + i * n_realizations + r):
                log = run(
                    exp_name=f's6_q{q_star}_rho{rho_cue}_r{r}',
                    true_k=q_star, fitness_gamma=FITNESS_GAMMA,
                    n=N, T=T, class_size=CLASS_SIZE, xi=SIGMA,
                    ks=KS, to_return=['class_frequency'],
                    cue_ar1_rho=rho_cue,
                )
            sel = KS[int(np.argmax(log['class_frequency'].mean(axis=0)))]
            bubble[i, KS.index(sel)] += 1.0 / n_realizations
    return bubble


def main():
    n_real = max(N_REALIZATIONS // 2, 50)
    bubbles = {}
    for label, rho in [('iid', 0.0), ('ar1_99', 0.99), ('constant', 1.0)]:
        print(f'Linear cues: {label}')
        bubbles[label] = linear_bubble(rho, n_real)
    np.savez(data_path('s6_cues_linear.npz'), ks=np.array(KS), **bubbles)
    print('Saved data/s6_cues_linear.npz')

    nn_bubbles = {}
    for label, rho in [('iid', 0.0), ('ar1_99', 0.99), ('constant', 1.0)]:
        print(f'NN cues: {label}')
        d = run_nn_experiment(
            true_ks=NN_KS, ks=NN_KS, class_size=CLASS_SIZE,
            T=T, fitness_gamma=0.01, xi=1.0,
            n_in=N, n_out=N,
            change_rate=0, cue_rho=rho,
            n_realizations=n_real,
        )
        nn_bubbles[label] = compute_bubble_matrix(d, NN_KS)
    np.savez(data_path('s6_cues_nn.npz'), ks=np.array(NN_KS), **nn_bubbles)
    print('Saved data/s6_cues_nn.npz')


if __name__ == '__main__':
    main()
