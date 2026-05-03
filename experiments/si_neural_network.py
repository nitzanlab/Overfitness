"""
SI Fig S2 (mean fitness curves) and Fig S4 (selected complexity heatmap)
for two-layer ReLU neural networks.

Note: SI Fig S4 caption uses gamma=0.01, not the global default (0.05).

Source: `evo_complexity_NN.ipynb` + `run_nn_correlated.py`.
"""
import numpy as np
from overfitness_paper.models.neural_net import (
    run_experiment, compute_bubble_matrix,
)
from overfitness_paper.config import T, N
from ._paths import data_path

NN_KS = [20, 50, 100, 500]
TRUE_KS = [20, 50, 100, 500]
CLASS_SIZE = 2000
GAMMA_NN = 0.01
XI = 1.0
N_REALIZATIONS = 100


def main():
    # S4A bubble: static environment, iid cues
    data_static = run_experiment(
        true_ks=TRUE_KS, ks=NN_KS, class_size=CLASS_SIZE,
        T=T, fitness_gamma=GAMMA_NN, xi=XI,
        n_in=N, n_out=N,
        change_rate=0, cue_rho=0, n_realizations=N_REALIZATIONS,
    )
    bubble = compute_bubble_matrix(data_static, NN_KS)

    # S4B heatmap over change rate
    change_rates = [0.0, 0.025, 0.05, 0.075, 0.1]
    heatmap = np.zeros((len(change_rates), len(TRUE_KS)))
    for i, cr in enumerate(change_rates):
        if cr == 0:
            d = data_static
        else:
            d = run_experiment(
                true_ks=TRUE_KS, ks=NN_KS, class_size=CLASS_SIZE,
                T=T, fitness_gamma=GAMMA_NN, xi=XI,
                n_in=N, n_out=N,
                change_rate=cr, cue_rho=0,
                n_realizations=N_REALIZATIONS,
            )
        # selected complexity = mean over realizations of class with highest final freq
        final = d[:, :, -1, :]
        sel_idx = np.argmax(final, axis=2)  # (n_true_ks, n_realizations)
        heatmap[i] = np.mean(np.array(NN_KS)[sel_idx], axis=1)

    np.savez(data_path('s4_nn.npz'),
             ks=np.array(NN_KS), true_ks=np.array(TRUE_KS),
             bubble=bubble, change_rates=np.array(change_rates),
             heatmap=heatmap)
    print('Saved data/s4_nn.npz')


if __name__ == '__main__':
    main()
