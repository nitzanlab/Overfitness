"""
Canonical default parameters used in the paper.

Centralizing these here resolves a discrepancy where some runner scripts
defaulted to fitness_gamma=0.1 while the published main results use 0.05.
Per-figure deviations (e.g. neural-network experiments using gamma=0.01,
mean-fitness curves using sigma=0.1 in captions vs sigma=1 in the code)
are documented in `data/README.md`.
"""

# Selection strength (eq. for f = exp(-gamma * ||...||))
FITNESS_GAMMA = 0.05

# Noise level on optimal phenotype (xi/sigma in code; sigma in paper)
SIGMA = 1.0

# Default simulation length
T = 1000

# Number of organisms per complexity class
CLASS_SIZE = 2000

# Input/output dimension for linear maps; complexity range is then [1, n*n].
N = 3

# Complexity values swept in main figures (linear maps with n=3 → q in 1..9).
KS = list(range(1, 10))

# Population for NN simulations uses different complexity values:
NN_KS = [20, 50, 100, 500]

# Number of realizations averaged per condition in main-text panels.
N_REALIZATIONS = 100

# Random seed used for "typical realization" Muller plots in Fig 1F, 3A, 4A, 4B.
DEMO_SEED = 0
