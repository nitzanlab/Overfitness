"""
SI Fig S7: selection accuracy as a function of class size.

Output: data/s7_class_size.npz

Source: extracted from `correlated_cues.ipynb` (the S7 panel was embedded there).
"""
import numpy as np
from overfitness_paper import run, np_temp_seed
from overfitness_paper.config import FITNESS_GAMMA, SIGMA, N, KS
from ._paths import data_path

CLASS_SIZES = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
T_S7 = 5000
N_REAL = 100


def main():
    accs = np.zeros(len(CLASS_SIZES))
    for i, cs in enumerate(CLASS_SIZES):
        match = 0
        total = 0
        for q_star in KS:
            for r in range(N_REAL):
                with np_temp_seed(70000 + i * 1000 + KS.index(q_star) * N_REAL + r):
                    log = run(
                        exp_name=f's7_cs{cs}_q{q_star}_r{r}',
                        true_k=q_star, fitness_gamma=FITNESS_GAMMA,
                        n=N, T=T_S7, class_size=cs, xi=SIGMA,
                        ks=KS, to_return=['class_frequency'],
                    )
                sel = KS[int(np.argmax(log['class_frequency'].mean(axis=0)))]
                match += (sel == q_star)
                total += 1
        accs[i] = match / total
        print(f'  class_size={cs}: accuracy={accs[i]:.3f}')
    np.savez(data_path('s7_class_size.npz'),
             class_sizes=np.array(CLASS_SIZES), accuracy=accs)
    print('Saved data/s7_class_size.npz')


if __name__ == '__main__':
    main()
