"""
SI Fig S14: parameter sensitivity. For each of (gamma, sigma, class_size, T),
sweep that parameter holding others at the canonical defaults and measure
selection accuracy (fraction of realizations where selected_q == q*).

Output: data/s14_sensitivity.npz

Source: sensitivity_analysis.ipynb §2.
"""
import numpy as np
from overfitness_paper import run, np_temp_seed
from overfitness_paper.config import (
    FITNESS_GAMMA, SIGMA, T, CLASS_SIZE, N, KS, N_REALIZATIONS,
)
from ._paths import data_path

GAMMAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SIGMAS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
CLASS_SIZES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
TS = [50, 100, 200, 500, 1000, 2000, 5000, 10000]


def selection_accuracy(gamma, sigma, class_size, T_):
    match = 0
    total = 0
    for q_star in KS:
        for r in range(N_REALIZATIONS):
            with np_temp_seed(120000 + q_star * 1000 + r):
                log = run(
                    exp_name=f's14_g{gamma}_s{sigma}_cs{class_size}_T{T_}_q{q_star}_r{r}',
                    true_k=q_star, fitness_gamma=gamma,
                    n=N, T=T_, class_size=class_size, xi=sigma,
                    ks=KS, to_return=['class_frequency'],
                )
            sel = KS[int(np.argmax(log['class_frequency'].mean(axis=0)))]
            match += (sel == q_star)
            total += 1
    return match / total


def main():
    accs = {}
    for label, sweep, defaults in [
        ('gamma', GAMMAS, dict(sigma=SIGMA, class_size=CLASS_SIZE, T_=T)),
        ('sigma', SIGMAS, dict(gamma=FITNESS_GAMMA, class_size=CLASS_SIZE, T_=T)),
        ('class_size', CLASS_SIZES, dict(gamma=FITNESS_GAMMA, sigma=SIGMA, T_=T)),
        ('T', TS, dict(gamma=FITNESS_GAMMA, sigma=SIGMA, class_size=CLASS_SIZE)),
    ]:
        accs[label + '_values'] = np.array(sweep)
        out = np.zeros(len(sweep))
        for i, v in enumerate(sweep):
            kwargs = dict(defaults)
            kwargs[label if label != 'T' else 'T_'] = v
            out[i] = selection_accuracy(**kwargs)
            print(f'  {label}={v}: acc={out[i]:.3f}')
        accs[label + '_accuracy'] = out
    np.savez(data_path('s14_sensitivity.npz'), **accs)
    print('Saved data/s14_sensitivity.npz')


if __name__ == '__main__':
    main()
