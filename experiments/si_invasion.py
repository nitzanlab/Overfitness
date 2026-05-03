"""
SI Fig S8: invasion experiments.

A small (1%) invader population at complexity q_inv is introduced into a
resident population at complexity q_res. The environment's true complexity is
fixed at the invader's value (q* = q_inv). We measure how often the invader
takes over.

Outputs:
- data/s8_invasion_change0.npy   : (n_ks, n_ks) success at change_rate=0
- data/s8_invasion_change02.npy  : (n_ks, n_ks) success at change_rate=0.2
- data/s8_mullers.pkl            : two single-realization Mullers (panels A, B)

Source: invasion_experiments.ipynb. Uses `run` with `init_frequency` to seed
the two-class composition (no separate invasion module function needed).
"""
import pickle
import numpy as np
from overfitness_paper import run, np_temp_seed
from overfitness_paper.simulation import setup_init_frequency
from overfitness_paper.config import (
    FITNESS_GAMMA, SIGMA, T, CLASS_SIZE, N, KS,
)
from ._paths import data_path

INVADER_FREQ = 0.01
N_REAL = 200
TAKEOVER_THRESHOLD = 0.5  # invader "wins" if its class freq exceeds this


def _two_class_init(invader_idx, resident_idx, n_ks, class_size):
    """Build init_frequency vector with invader at 1% and resident at 99%."""
    class_freqs = np.zeros(n_ks)
    class_freqs[resident_idx] = 1 - INVADER_FREQ
    class_freqs[invader_idx] = INVADER_FREQ
    return setup_init_frequency(class_freqs, [class_size] * n_ks)


def invasion_success(q_inv, q_res, change_rate, n_realizations):
    n_ks = len(KS)
    inv_idx = KS.index(q_inv)
    res_idx = KS.index(q_res)
    init = _two_class_init(inv_idx, res_idx, n_ks, CLASS_SIZE)

    n_envs = max(1, int(round(change_rate * T))) if change_rate > 0 else 1
    wins = 0
    for r in range(n_realizations):
        with np_temp_seed(80000 + q_inv * 1000 + q_res * 100 + r):
            log = run(
                exp_name=f's8_inv{q_inv}_res{q_res}_cr{change_rate}_r{r}',
                true_k=q_inv, fitness_gamma=FITNESS_GAMMA,
                n=N, T=T, class_size=CLASS_SIZE, xi=SIGMA,
                ks=KS, to_return=['class_frequency'], n_envs=n_envs,
                init_frequency=init,
            )
        final = log['class_frequency'][-1]
        if final[inv_idx] >= TAKEOVER_THRESHOLD:
            wins += 1
    return wins / n_realizations


def main():
    n_ks = len(KS)
    for label, change_rate in [('change0', 0.0), ('change02', 0.2)]:
        print(f'Invasion sweep: {label}')
        success = np.zeros((n_ks, n_ks))
        for i, q_inv in enumerate(KS):
            for j, q_res in enumerate(KS):
                if q_inv == q_res:
                    success[i, j] = np.nan
                    continue
                success[i, j] = invasion_success(q_inv, q_res, change_rate, N_REAL)
                print(f'  q_inv={q_inv}, q_res={q_res}: success={success[i, j]:.2f}')
        np.save(data_path(f's8_invasion_{label}.npy'), success)
        print(f'  saved data/s8_invasion_{label}.npy')

    # Two example Mullers (panel A: q*=q_inv=3 invading q_res=7; B: q*=q_inv=7 invading q_res=3)
    mullers = {}
    for label, q_inv, q_res in [('q3', 3, 7), ('q7', 7, 3)]:
        n_ks = len(KS)
        init = _two_class_init(KS.index(q_inv), KS.index(q_res), n_ks, CLASS_SIZE)
        with np_temp_seed(80000 + q_inv):
            log = run(
                exp_name=f's8_muller_{label}',
                true_k=q_inv, fitness_gamma=FITNESS_GAMMA,
                n=N, T=50, class_size=CLASS_SIZE, xi=SIGMA,
                ks=KS, to_return=['class_frequency'], init_frequency=init,
            )
        mullers[label] = log
    with open(data_path('s8_mullers.pkl'), 'wb') as f:
        pickle.dump(mullers, f)
    print('Saved data/s8_mullers.pkl')


if __name__ == '__main__':
    main()
