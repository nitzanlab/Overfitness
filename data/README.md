# `data/` — cached simulation outputs

This directory ships **empty**. Populating it requires running the matching
script in `../experiments/` (see top-level `README.md`).

## Filename → figure mapping

| File | Produced by | Used by |
|---|---|---|
| `fig2_bubbles.npz`            | `main_fig2_stable_env.py`    | Fig 2A, 2B |
| `fig2_dynamics_q5.npz`        | `main_fig2_stable_env.py`    | Fig 2C–F |
| `fig3_dynamics.npz`           | `main_fig3_transients.py`    | Fig 3C–F |
| `fig3_fitness_dist.npz`       | `main_fig3_transients.py`    | Fig 3B |
| `fig4_mullers.pkl`            | `main_fig4_env_change.py`    | Fig 4A, 4B |
| `fig4_dynamics.npz`           | `main_fig4_env_change.py`    | Fig 4C, 4D |
| `fig4e_heatmap.npz`           | `main_fig4_env_change.py`    | Fig 4E |
| `s1_mean_fitness.npz`         | `si_mean_fitness_curves.py`  | Fig S1 |
| `s2_mean_fitness_nn.npz`      | `si_neural_network.py`       | Fig S2 |
| `s3_polynomial.npz`           | `si_polynomial.py`           | Fig S3 |
| `s4_nn.npz`                   | `si_neural_network.py`       | Fig S4 |
| `s5_ar1_linear.npz`           | `si_ar1_environments.py`     | Fig S5 (top) |
| `s5_ar1_nn.npz`               | `si_ar1_environments.py`     | Fig S5 (bottom) |
| `s6_cues_linear.npz`          | `si_correlated_cues.py`      | Fig S6 (top) |
| `s6_cues_nn.npz`              | `si_correlated_cues.py`      | Fig S6 (bottom) |
| `s7_class_size.npz`           | `si_class_size_sweep.py`     | Fig S7 |
| `s8_invasion_change0.npy`     | `si_invasion.py`             | Fig S8C |
| `s8_invasion_change02.npy`    | `si_invasion.py`             | Fig S8D |
| `s8_mullers.pkl`              | `si_invasion.py`             | Fig S8A, S8B |
| `s9_q1.npz`, `s9_q9.npz`      | `si_mutations.py`            | Fig S9A–C |
| `s9_grid.npz`                 | `si_mutations.py`            | Fig S9D |
| `s10_alt_fitness.npz`         | `si_alt_fitness.py`          | Fig S10 |
| `s11_alt_fitness_change.npz`  | `si_alt_fitness.py`          | Fig S11 |
| `s12_alt_noise.npz`           | `si_alt_noise.py`            | Fig S12 |
| `s13_alt_noise_change.npz`    | `si_alt_noise.py`            | Fig S13 |
| `s14_sensitivity.npz`         | `si_param_sensitivity.py`    | Fig S14 |

## Inconsistencies flagged during migration

(Cross-references to §4 of the migration plan.)

### 4.1 Two parallel simulation engines (resolved)
The original repo had `bayesian_evolution.py` and `overfitness.py` as two
independent simulation engines. SI Fig S3, S4, S8 used the former; everything
else used the latter. Here we use only `overfitness_paper.simulation`
(adapted from `overfitness.py`).

### 4.2 `scipy` missing from requirements (resolved)
Added.

### 4.3 `torch` missing from requirements (resolved)
Added (used by NN experiments).

### 4.4 Default `fitness_gamma` differs across files (resolved)
The published paper uses γ=0.05. The legacy `run_linear_correlated.py` had
argparse default 0.1. All experiment scripts in this directory now read
`overfitness_paper.config.FITNESS_GAMMA = 0.05`. The exception is the NN
simulations (Fig S2, S4, S5D-F, S6D-F) which use γ=0.01 per the SI caption;
this override is hard-coded in `si_neural_network.py`.

### 4.5 σ inconsistency in S1/S2 (flagged, NOT fixed)
SI Fig S1/S2 captions state σ=0.1. The source notebook
(`mean_fitness_vs_complexity.ipynb`) does not set σ explicitly and therefore
uses the code default σ=1.0. **Per author direction we keep σ=1.0 and flag
this caption for the manuscript update.**

### 4.6 Stale figure numbering in source notebook (resolved)
The legacy `reproduce_figures.ipynb` referred to "Fig 5/6/7/8" which are
now SI Fig S3/S4/S7/S8. The new notebooks use the current numbering.

### 4.7 Missing main-figure outputs (resolved)
The legacy `paper_figures/` folder shipped only Fig 4A and Fig 4B; Fig 4C,
4D, 4E were never re-saved. The new `main_figures.ipynb` saves all panels.

### 4.8 Population size in Fig S7 (resolved)
The S7 panel was embedded inside `correlated_cues.ipynb`. Now lives in its
own script `si_class_size_sweep.py`.

### 4.9 NN code lived only in runner script (resolved)
The 2-layer NN model was defined inside `run_nn_correlated.py` and not
exposed as library code. Now in `src/overfitness_paper/models/neural_net.py`.
