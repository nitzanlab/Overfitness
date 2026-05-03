# Fitness and Overfitness: Implicit Regularization in Evolutionary Dynamics

This repository contains code to reproduce simulations and generate plots from the manuscript:

**Fitness and Overfitness: Implicit Regularization in Evolutionary Dynamics**  
Hagai Rappeport and Mor Nitzan  


---

## Overview

This project implements a computational framework to study the evolution of biological complexity through the lens of **implicit regularization** in evolutionary dynamics. It leverages the mathematical analogy between the replicator equation and Bayesian inference to explore how organismal complexity evolves to match environmental complexity.


![Simulation results](figures/overfitness_github_fig.png "Selection dynamics of an optimal replicator/hypothesis")


---

## Repository Contents

```
src/overfitness_paper/      # Python package (importable as `overfitness_paper`)
  simulation.py             # replicator dynamics + linear/polynomial classes
  invasion.py               # invasion-tensor analysis
  models/neural_net.py      # 2-layer ReLU NN model used in SI figures
  metrics_helpers.py        # run_with_metrics() — wraps run() + metric helpers
  config.py                 # canonical default parameters

experiments/                # one script per figure, saves to data/
  main_fig{2,3,4}_*.py      # main-text figures
  si_*.py                   # supplementary figures S1–S14

data/                       # cached simulation outputs (.npz / .pkl)
                            # ships empty; populated by experiments/

notebooks/
  main_figures.ipynb        # generates Fig 1–4 from data/
  supplementary_figures.ipynb  # generates Fig S1–S14 from data/

figures/                    # rendered PDFs
  main/
  supplementary/
```

---

## Reproducing the figures

Install dependencies:

```bash
pip install -r requirements.txt
```

For each figure, first regenerate the cached simulation output, then run the corresponding notebook cell:

```bash
# Main figures
python -m experiments.main_fig2_stable_env
python -m experiments.main_fig3_transients
python -m experiments.main_fig4_env_change
jupyter nbconvert --to notebook --execute notebooks/main_figures.ipynb

# Supplementary figures
python -m experiments.si_mean_fitness_curves    # S1
python -m experiments.si_neural_network         # S2, S4
python -m experiments.si_polynomial             # S3
python -m experiments.si_ar1_environments       # S5
python -m experiments.si_correlated_cues        # S6
python -m experiments.si_class_size_sweep       # S7
python -m experiments.si_invasion               # S8
python -m experiments.si_mutations              # S9
python -m experiments.si_alt_fitness            # S10, S11
python -m experiments.si_alt_noise              # S12, S13
python -m experiments.si_param_sensitivity      # S14
jupyter nbconvert --to notebook --execute notebooks/supplementary_figures.ipynb
```

Each script is independent and idempotent. Wall-clock per script ranges from seconds to several hours depending on the size of the parameter sweep. All scripts read defaults from `overfitness_paper.config`; figure-specific overrides (e.g. γ=0.01 for the NN simulations) are documented inline.

See `data/README.md` for a filename → figure mapping.

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Seaborn
- SciPy
- PyTorch (for the neural-network experiments)
- Jupyter
- tqdm

---

## License
This project is released under the MIT License. See LICENSE for details.

## Contact
For questions or collaboration inquiries, please contact:

Hagai Rappeport — [hagai.rappeport@huji.mail.ac.il]
Mor Nitzan — [mor.nitzan@huji.mail.ac.il]
