"""
Overfitness: Implicit Regularization in Evolutionary Dynamics
=============================================================

Core simulation and visualization module for reproducing paper figures.

This module implements the replicator equation dynamics for studying how
organismal complexity evolves to match environmental complexity through
implicit regularization mechanisms.

Reference:
    Rappeport & Nitzan (2026). "Fitness and Overfitness: Implicit Regularization
    in Evolutionary Dynamics"
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import contextlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# SEED MANAGEMENT
# =============================================================================

@contextlib.contextmanager
def np_temp_seed(temp_seed):
    """
    Context manager for temporary numpy random seed.

    Allows running code with a specific seed while preserving
    the original random state afterwards.

    Parameters
    ----------
    temp_seed : int or None
        Seed to use. If None, no seed is set.

    Example
    -------
    >>> with np_temp_seed(42):
    ...     result = np.random.rand()
    """
    if temp_seed is None:
        yield
    else:
        original_state = np.random.get_state()
        np.random.seed(temp_seed)
        try:
            yield
        finally:
            np.random.set_state(original_state)

# =============================================================================
# UTILITIES
# =============================================================================

def chunk_sum_reduceat(X, K):
    """
    Sum chunks of array X with sizes indicated by K using np.add.reduceat.

    Parameters
    ----------
    X : array
        Array to sum in chunks
    K : array-like
        Chunk sizes

    Returns
    -------
    array
        Summed chunks
    """
    indices = np.concatenate([[0], np.cumsum(K)[:-1]])
    return np.add.reduceat(X, indices)


def get_idx(n, k):
    """
    Generate indices for k non-zero entries in an n×n matrix.

    Uses a deterministic pattern filling diagonal first, then off-diagonals.

    Parameters
    ----------
    n : int
        Matrix dimension
    k : int
        Number of non-zero entries

    Returns
    -------
    array (k, 2)
        Row, column indices for non-zero entries
    """
    idx = np.zeros(shape=(k, 2), dtype=int)
    for i in range(k):
        idx[i, :] = [(i + i//n) % n, i % n]
    return idx


def setup_init_frequency(class_freqs, class_size_list):
    """
    Setup initial frequency distribution for invasion experiments.

    Parameters
    ----------
    class_freqs : array-like
        Target frequency for each class (must sum to 1)
    class_size_list : array-like
        Number of individuals in each class

    Returns
    -------
    array
        Initial frequency vector for all individuals
    """
    assert len(class_freqs) == len(class_size_list), \
        "class_freqs and class_size_list must have the same length"
    assert np.isclose(np.sum(class_freqs), 1), "class_freqs must sum to 1"

    indices = np.concatenate([[0], np.cumsum(class_size_list)])
    init_frequency = np.zeros(sum(class_size_list))

    for class_i, class_freq in enumerate(class_freqs):
        for j in range(indices[class_i], indices[class_i + 1]):
            init_frequency[j] = class_freq / class_size_list[class_i]

    return init_frequency


# =============================================================================
# MUTATION FUNCTIONS
# =============================================================================

def get_complexity_from_matrix(A, n):
    """
    Determine the complexity class of a matrix.

    Complexity is the number of non-zero entries at the standard positions
    defined by get_idx(). Checks positions in reverse order to find the
    highest occupied position.

    Parameters
    ----------
    A : array (n, n)
        Linear map matrix
    n : int
        Matrix dimension

    Returns
    -------
    int
        Complexity class k (number of non-zero entries, minimum 1)
    """
    max_k = n * n
    for k in range(max_k, 0, -1):
        idx = get_idx(n, k)
        last_pos = (idx[-1, 0], idx[-1, 1])
        if np.abs(A[last_pos]) > 1e-10:
            return k
    return 1  # Minimum complexity is 1


def mutate_demote(A, n, k):
    """
    Demote a matrix from complexity k to k-1 by zeroing the last parameter.

    Parameters
    ----------
    A : array (n, n)
        Linear map matrix with complexity k
    n : int
        Matrix dimension
    k : int
        Current complexity (must be > 1)

    Returns
    -------
    array (n, n)
        New matrix with complexity k-1
    """
    if k <= 1:
        raise ValueError("Cannot demote from complexity 1")

    A_new = A.copy()
    idx = get_idx(n, k)
    last_pos = (idx[-1, 0], idx[-1, 1])
    A_new[last_pos] = 0.0
    return A_new


def mutate_promote(A, n, k, mean=0, std=1):
    """
    Promote a matrix from complexity k to k+1 by adding a new parameter.

    The new parameter is sampled from N(mean, std) and placed at the
    (k+1)-th position according to the get_idx() pattern.

    Parameters
    ----------
    A : array (n, n)
        Linear map matrix with complexity k
    n : int
        Matrix dimension
    k : int
        Current complexity (must be < n*n)
    mean : float
        Mean of normal distribution for new entry
    std : float
        Standard deviation of normal distribution

    Returns
    -------
    array (n, n)
        New matrix with complexity k+1
    """
    max_k = n * n
    if k >= max_k:
        raise ValueError(f"Cannot promote beyond complexity {max_k}")

    A_new = A.copy()
    idx = get_idx(n, k + 1)
    new_pos = (idx[-1, 0], idx[-1, 1])
    A_new[new_pos] = np.random.normal(mean, std)
    return A_new


def mutate_within_class(A, n, k, mean=0, std=1):
    """
    Create a within-class mutant by resampling all k parameters.

    Keeps the same complexity (same positions non-zero) but
    redraws all k values from N(mean, std).

    Parameters
    ----------
    A : array (n, n)
        Linear map matrix with complexity k (not used, but kept for API consistency)
    n : int
        Matrix dimension
    k : int
        Current complexity (number of non-zero parameters)
    mean : float
        Mean of normal distribution for parameter values
    std : float
        Standard deviation of normal distribution

    Returns
    -------
    array (n, n)
        New matrix with same complexity k but resampled parameter values
    """
    A_new = np.zeros((n, n))
    idx = get_idx(n, k)
    for i in range(k):
        pos = (idx[i, 0], idx[i, 1])
        A_new[pos] = np.random.normal(mean, std)
    return A_new


def compute_avg_complexity(class_freqs, ks):
    """
    Compute weighted average complexity from class frequencies.

    Parameters
    ----------
    class_freqs : array (n_classes,)
        Frequency of each complexity class (should sum to 1)
    ks : list or array
        Complexity values for each class

    Returns
    -------
    float
        Expected complexity E[k] = sum(freq_k * k)
    """
    return np.sum(np.array(class_freqs) * np.array(ks))


# =============================================================================
# DYNAMIC POPULATION FOR MUTATION SIMULATIONS
# =============================================================================

class MutationPopulation:
    """
    Dynamic population that can grow/shrink via mutation and extinction.

    This class manages a population of genotypes (types) where:
    - Types can be added via mutation
    - Types with frequency < 1/M are removed (extinction)
    - Each type has a unique ID for lineage tracking

    Parameters
    ----------
    n : int
        Matrix dimension for linear maps
    M : int
        Number of individuals (sets frequency discretization)
    ks : list
        List of all possible complexity classes

    Attributes
    ----------
    types : list of arrays
        Each entry is a genotype matrix (n, n)
    frequencies : np.ndarray
        Current frequency of each type (sums to 1)
    complexities : np.ndarray
        Complexity class of each type
    type_ids : np.ndarray
        Unique ID for each type (for lineage tracking)
    birth_times : np.ndarray
        Timestep when each type was created (-1 for initial types)
    """

    def __init__(self, n, M, ks):
        self.n = n
        self.M = M
        self.ks = ks
        self.types = []
        self.frequencies = np.array([])
        self.complexities = np.array([], dtype=int)
        self.type_ids = np.array([], dtype=int)
        self.birth_times = np.array([], dtype=int)
        self._next_type_id = 0

    def __len__(self):
        return len(self.types)

    def initialize_from_class(self, initial_k, class_size):
        """
        Initialize population with types all in one complexity class.

        Parameters
        ----------
        initial_k : int
            Starting complexity class
        class_size : int
            Number of types to create
        """
        self.types = []
        for _ in range(class_size):
            A = sample_parameters_linear(self.n, initial_k)
            self.types.append(A)

        n_types = len(self.types)
        self.frequencies = np.ones(n_types) / n_types
        self.complexities = np.full(n_types, initial_k, dtype=int)
        self.type_ids = np.arange(n_types, dtype=int)
        self.birth_times = np.full(n_types, -1, dtype=int)  # -1 = initial
        self._next_type_id = n_types

    def get_types_array(self):
        """
        Get population as a single numpy array for fitness computation.

        Returns
        -------
        np.ndarray (n_types, n, n)
            Stacked array of all type matrices
        """
        if len(self.types) == 0:
            return np.array([]).reshape(0, self.n, self.n)
        return np.stack(self.types, axis=0)

    def add_mutant(self, matrix, complexity, birth_time):
        """
        Add a new mutant type with frequency 1/M.

        Existing frequencies are scaled by (1 - 1/M) to maintain sum = 1.

        Parameters
        ----------
        matrix : array (n, n)
            The mutant genotype matrix
        complexity : int
            Complexity class of the mutant
        birth_time : int
            Timestep when mutation occurred

        Returns
        -------
        int
            The unique type_id assigned to this mutant
        """
        # Scale existing frequencies
        scale_factor = 1.0 - 1.0 / self.M
        self.frequencies = self.frequencies * scale_factor

        # Add new mutant
        self.types.append(matrix.copy())
        self.frequencies = np.append(self.frequencies, 1.0 / self.M)
        self.complexities = np.append(self.complexities, complexity)
        self.birth_times = np.append(self.birth_times, birth_time)

        type_id = self._next_type_id
        self.type_ids = np.append(self.type_ids, type_id)
        self._next_type_id += 1

        return type_id

    def remove_extinct(self):
        """
        Remove types with frequency < 1/M.

        Returns
        -------
        list of int
            Type IDs that were removed
        """
        threshold = 1.0 / self.M
        alive_mask = self.frequencies >= threshold

        if np.all(alive_mask):
            return []

        extinct_ids = self.type_ids[~alive_mask].tolist()

        # Keep only alive types
        self.types = [self.types[i] for i in range(len(self.types)) if alive_mask[i]]
        self.frequencies = self.frequencies[alive_mask]
        self.complexities = self.complexities[alive_mask]
        self.type_ids = self.type_ids[alive_mask]
        self.birth_times = self.birth_times[alive_mask]

        # Renormalize frequencies
        if len(self.frequencies) > 0:
            self.frequencies = self.frequencies / self.frequencies.sum()

        return extinct_ids

    def get_class_frequencies(self):
        """
        Sum frequencies by complexity class.

        Returns
        -------
        np.ndarray (len(ks),)
            Frequency of each complexity class
        """
        class_freqs = np.zeros(len(self.ks))
        for i, k in enumerate(self.ks):
            mask = self.complexities == k
            class_freqs[i] = np.sum(self.frequencies[mask])
        return class_freqs

    def get_type_idx_by_id(self, type_id):
        """
        Get the current index of a type by its ID.

        Parameters
        ----------
        type_id : int
            Unique type identifier

        Returns
        -------
        int or None
            Current index in arrays, or None if extinct
        """
        matches = np.where(self.type_ids == type_id)[0]
        if len(matches) == 0:
            return None
        return matches[0]


class LineageTracker:
    """
    Track mutant lineages to detect successful invasions.

    A lineage is tracked from birth until either:
    - It reaches the invasion threshold frequency (successful invasion)
    - It goes extinct (frequency < 1/M, removed from population)

    Parameters
    ----------
    invasion_threshold : float
        Frequency threshold for "takeover" (default 0.5)
    """

    def __init__(self, invasion_threshold=0.5):
        self.invasion_threshold = invasion_threshold
        self.active_lineages = {}  # type_id -> lineage_info
        self.successful_invasions = []
        self.failed_lineages = []

    def register_mutant(self, type_id, birth_time, birth_class, mutation_type='between'):
        """
        Start tracking a new mutant lineage.

        Parameters
        ----------
        type_id : int
            Unique identifier for the mutant type
        birth_time : int
            Timestep when mutation occurred
        birth_class : int
            Complexity class the mutant was born into
        mutation_type : str
            Type of mutation: 'between' (promote/demote) or 'within' (resample)
        """
        self.active_lineages[type_id] = {
            'birth_time': birth_time,
            'birth_class': birth_class,
            'mutation_type': mutation_type,
            'peak_freq': 0.0
        }

    def update(self, t, population):
        """
        Update all active lineages based on current population state.

        Checks for:
        - Takeovers (frequency >= threshold)
        - Extinctions (type no longer in population)

        Parameters
        ----------
        t : int
            Current timestep
        population : MutationPopulation
            Current population state
        """
        current_type_ids = set(population.type_ids)

        for type_id, info in list(self.active_lineages.items()):
            if type_id not in current_type_ids:
                # Lineage went extinct
                self.failed_lineages.append({
                    'type_id': type_id,
                    'birth_time': info['birth_time'],
                    'birth_class': info['birth_class'],
                    'mutation_type': info.get('mutation_type', 'between'),
                    'extinction_time': t,
                    'peak_freq': info['peak_freq']
                })
                del self.active_lineages[type_id]
            else:
                # Update frequency
                idx = population.get_type_idx_by_id(type_id)
                freq = population.frequencies[idx]
                info['peak_freq'] = max(info['peak_freq'], freq)

                if freq >= self.invasion_threshold:
                    # Successful invasion!
                    self.successful_invasions.append({
                        'type_id': type_id,
                        'birth_time': info['birth_time'],
                        'takeover_time': t,
                        'wait_time': t - info['birth_time'],
                        'birth_class': info['birth_class'],
                        'mutation_type': info.get('mutation_type', 'between'),
                        'final_freq': freq
                    })
                    del self.active_lineages[type_id]

    def get_invasion_events(self):
        """
        Get list of successful invasion events.

        Returns
        -------
        list of dict
            Each dict contains: type_id, birth_time, takeover_time,
            wait_time, birth_class, final_freq
        """
        return self.successful_invasions.copy()

    def get_invasion_times_by_class(self):
        """
        Get invasion times grouped by birth complexity class.

        Returns
        -------
        dict
            Maps complexity class -> list of (birth_time, takeover_time)
        """
        by_class = {}
        for inv in self.successful_invasions:
            k = inv['birth_class']
            if k not in by_class:
                by_class[k] = []
            by_class[k].append((inv['birth_time'], inv['takeover_time']))
        return by_class


# =============================================================================
# FUNCTION CLASSES - LINEAR
# =============================================================================

def sample_parameters_linear(n, k, mean=0, std=1):
    """
    Sample an n×n matrix with k non-zero entries.

    Complexity is defined as the number of non-zero entries.

    Parameters
    ----------
    n : int
        Matrix dimension
    k : int
        Number of non-zero entries (complexity)
    mean : float
        Mean of normal distribution for entries
    std : float
        Standard deviation of normal distribution

    Returns
    -------
    array (n, n)
        Sparse matrix with k non-zero entries
    """
    A = np.zeros(shape=(n, n))
    idx = get_idx(n, k)
    entries = np.random.normal(mean, std, size=k)
    A[idx[:, 0], idx[:, 1]] = entries
    return A


def fitness_single_linear(A, E, gamma=1):
    """
    Calculate fitness for a single linear map organism.

    Parameters
    ----------
    A : array (n, n)
        Linear map matrix
    E : array (2, n)
        Environment: E[0] is input signal, E[1] is optimal response
    gamma : float
        Selection strength

    Returns
    -------
    float
        Fitness value
    """
    return np.exp(-gamma * np.linalg.norm(A @ E[0] - E[1]))


def fitness_population_linear(population, E, gamma=1):
    """
    Calculate fitness for a population of linear maps.

    Parameters
    ----------
    population : array (N, n, n)
        Population of N linear map matrices
    E : array (2, n)
        Environment: E[0] is input signal, E[1] is optimal response
    gamma : float
        Selection strength

    Returns
    -------
    array (N,)
        Fitness values for each organism
    """
    return np.exp(-gamma * np.linalg.norm(population @ E[0].T - E[1], axis=1))


def _generate_input_signals(T, n, cue_ar1_rho=None):
    """
    Generate T input signals of dimension n.

    If cue_ar1_rho is None or 0, inputs are iid N(0, I).
    Otherwise, inputs follow an AR(1) process:
        ξ(t+1) = ρ * ξ(t) + √(1 - ρ²) * ε(t),  ε ~ N(0, I)

    The process is stationary with marginal N(0, I) and
    Corr(ξ(t), ξ(t+k)) = ρ^k.

    Parameters
    ----------
    T : int
        Number of timesteps
    n : int
        Dimension
    cue_ar1_rho : float, optional
        AR(1) autocorrelation for input signals. 0 or None = iid.

    Returns
    -------
    array (T, n)
        Input signals
    """
    if cue_ar1_rho is not None and cue_ar1_rho > 0:
        signals = np.zeros((T, n))
        sigma = np.sqrt(1 - cue_ar1_rho**2)
        signals[0] = np.random.randn(n)
        for t in range(1, T):
            signals[t] = cue_ar1_rho * signals[t - 1] + sigma * np.random.randn(n)
        return signals
    else:
        return np.random.randn(T, n)


def sample_environments_linear(T, A, n, xi=0.1, cue_ar1_rho=None):
    """
    Sample T environments for the linear function class.

    Parameters
    ----------
    T : int
        Number of environments/timesteps
    A : array (n, n)
        Ground truth linear map
    n : int
        Dimension
    xi : float
        Noise level (sigma in paper)
    cue_ar1_rho : float, optional
        AR(1) autocorrelation for input signals. None or 0 = iid.

    Returns
    -------
    array (T, 2, n)
        T environments, each with input signal and optimal response
    """
    E = np.zeros((T, 2, n))
    E[:, 0, :] = _generate_input_signals(T, n, cue_ar1_rho)
    E[:, 1, :] = (A @ E[:, 0, :].T).T
    E[:, 1, :] += xi * np.random.randn(T, n)
    return E


# =============================================================================
# ALTERNATIVE FITNESS FUNCTIONS (for sensitivity analysis)
# =============================================================================

def fitness_population_linear_L1(population, E, gamma=1):
    """
    L1 norm fitness: f = exp(-γ * ||φ(ξ) - φ*(ξ)||_1)

    Uses Manhattan distance instead of Euclidean.
    """
    errors = population @ E[0].T - E[1]
    return np.exp(-gamma * np.sum(np.abs(errors), axis=1))


def fitness_population_linear_rational(population, E, gamma=1):
    """
    Rational fitness: f = 1 / (1 + γ * ||φ(ξ) - φ*(ξ)||²)

    Bounded fitness function with softer tails than exponential.
    """
    errors = np.linalg.norm(population @ E[0].T - E[1], axis=1)
    return 1.0 / (1.0 + gamma * errors**2)


def fitness_population_linear_cutoff(population, E, gamma=1):
    """
    Linear cutoff fitness: f = max(0, 1 - γ * ||φ(ξ) - φ*(ξ)||)

    Linear decay with hard cutoff at zero.
    """
    errors = np.linalg.norm(population @ E[0].T - E[1], axis=1)
    return np.maximum(0, 1 - gamma * errors)


# =============================================================================
# ALTERNATIVE NOISE TYPES (for sensitivity analysis)
# =============================================================================

def sample_environments_linear_uniform(T, A, n, xi=0.1, cue_ar1_rho=None):
    """
    Uniform noise: ε ~ Uniform(-√3*σ, √3*σ)

    Same variance as Gaussian with std=σ.
    """
    E = np.zeros((T, 2, n))
    E[:, 0, :] = _generate_input_signals(T, n, cue_ar1_rho)
    E[:, 1, :] = (A @ E[:, 0, :].T).T
    # Uniform with same variance as N(0, xi²): range = [-√3*xi, √3*xi]
    E[:, 1, :] += np.random.uniform(-np.sqrt(3) * xi, np.sqrt(3) * xi, size=(T, n))
    return E


def sample_environments_linear_laplacian(T, A, n, xi=0.1, cue_ar1_rho=None):
    """
    Laplacian noise: ε ~ Laplace(0, σ/√2)

    Same variance as Gaussian with std=σ, but heavier tails.
    """
    E = np.zeros((T, 2, n))
    E[:, 0, :] = _generate_input_signals(T, n, cue_ar1_rho)
    E[:, 1, :] = (A @ E[:, 0, :].T).T
    # Laplace with same variance as N(0, xi²): scale = xi/√2
    E[:, 1, :] += np.random.laplace(0, xi / np.sqrt(2), size=(T, n))
    return E


# =============================================================================
# FITNESS AND NOISE TYPE SELECTORS (for sensitivity analysis)
# =============================================================================

# Store alternative fitness functions
_FITNESS_FUNCTIONS = {
    'L2': fitness_population_linear,  # default (Gaussian/exponential)
    'L1': fitness_population_linear_L1,
    'rational': fitness_population_linear_rational,
    'cutoff': fitness_population_linear_cutoff,
}

# Store alternative noise types
_NOISE_TYPES = {
    'gaussian': sample_environments_linear,  # default
    'uniform': sample_environments_linear_uniform,
    'laplacian': sample_environments_linear_laplacian,
}

# Current selections (used by set_fitness_type and set_noise_type)
_current_fitness_type = 'L2'
_current_noise_type = 'gaussian'


def set_fitness_type(fitness_type):
    """
    Set the fitness function type for linear simulations.

    Parameters
    ----------
    fitness_type : str
        One of: 'L2' (default), 'L1', 'rational', 'cutoff'
    """
    global fitness_population, _current_fitness_type
    if fitness_type not in _FITNESS_FUNCTIONS:
        raise ValueError(f"Unknown fitness type: {fitness_type}. "
                        f"Available: {list(_FITNESS_FUNCTIONS.keys())}")
    _current_fitness_type = fitness_type
    # Only update if currently using linear
    if sample_parameters == sample_parameters_linear:
        fitness_population = _FITNESS_FUNCTIONS[fitness_type]


def set_noise_type(noise_type):
    """
    Set the noise type for linear simulations.

    Parameters
    ----------
    noise_type : str
        One of: 'gaussian' (default), 'uniform', 'laplacian'
    """
    global sample_environments, _current_noise_type
    if noise_type not in _NOISE_TYPES:
        raise ValueError(f"Unknown noise type: {noise_type}. "
                        f"Available: {list(_NOISE_TYPES.keys())}")
    _current_noise_type = noise_type
    # Only update if currently using linear
    if sample_parameters == sample_parameters_linear:
        sample_environments = _NOISE_TYPES[noise_type]


def get_current_fitness_type():
    """Return the current fitness type."""
    return _current_fitness_type


def get_current_noise_type():
    """Return the current noise type."""
    return _current_noise_type


def reset_to_defaults():
    """Reset fitness and noise to defaults (L2/Gaussian)."""
    set_fitness_type('L2')
    set_noise_type('gaussian')


# =============================================================================
# FUNCTION CLASSES - POLYNOMIAL
# =============================================================================

def sample_parameters_polynomial(n, k, mean=0, std=1):
    """
    Sample polynomial coefficients with degree k.

    Complexity is defined as the polynomial degree.

    Parameters
    ----------
    n : int
        Maximum degree + 1 (number of coefficients)
    k : int
        Degree of polynomial (complexity)
    mean : float
        Mean of normal distribution for coefficients
    std : float
        Standard deviation

    Returns
    -------
    array (n,)
        Polynomial coefficients (first k are non-zero)
    """
    A = np.zeros(n)
    A[:k] = np.random.normal(mean, std, size=k)
    return A


def fitness_single_polynomial(A, E, gamma=1):
    """
    Calculate fitness for a single polynomial organism.

    Parameters
    ----------
    A : array (n,)
        Polynomial coefficients
    E : array (2, n)
        Environment: E[0] is input, E[1] is optimal output
    gamma : float
        Selection strength

    Returns
    -------
    float
        Fitness value
    """
    V = np.vander(E[0], increasing=True)
    return np.exp(-gamma * np.linalg.norm(V @ A - E[1]))


def fitness_population_polynomial(population, E, gamma=1):
    """
    Calculate fitness for a population of polynomials.

    Parameters
    ----------
    population : array (N, n)
        Population of N polynomial coefficient vectors
    E : array (2, n)
        Environment
    gamma : float
        Selection strength

    Returns
    -------
    array (N,)
        Fitness values
    """
    V = np.vander(E[0], increasing=True)
    E_target = E[1].reshape(-1, 1)
    fitness_values = np.exp(-gamma * np.linalg.norm(V @ population.T - E_target, axis=0))
    return fitness_values


def sample_environments_polynomial(T, A, n, xi=0.1):
    """
    Sample T environments for polynomial function class.

    Parameters
    ----------
    T : int
        Number of environments
    A : array (n,)
        Ground truth polynomial coefficients
    n : int
        Dimension
    xi : float
        Noise level

    Returns
    -------
    array (T, 2, n)
        T environments
    """
    E = np.zeros((T, 2, n))
    E[:, 0, :] = np.random.randn(T, n)
    E[:, 1, :] = np.polyval(A[::-1], E[:, 0, :].T).T
    E[:, 1, :] += xi * np.random.randn(T, n)
    return E


# =============================================================================
# FUNCTION TYPE SELECTOR (Global State)
# =============================================================================

# Global function pointers (set by set_function_type)
sample_parameters = sample_parameters_linear
fitness_single = fitness_single_linear
fitness_population = fitness_population_linear
sample_environments = sample_environments_linear


def set_function_type(setting):
    """
    Set the global function type for simulations.

    Parameters
    ----------
    setting : str
        'linear' or 'polynomial'

    Note: For linear, respects current fitness_type and noise_type settings.
    """
    global sample_parameters, fitness_single, fitness_population, sample_environments

    if setting == "polynomial":
        sample_parameters = sample_parameters_polynomial
        fitness_single = fitness_single_polynomial
        fitness_population = fitness_population_polynomial
        sample_environments = sample_environments_polynomial
    elif setting == "linear":
        sample_parameters = sample_parameters_linear
        fitness_single = fitness_single_linear
        # Use current fitness and noise type selections
        fitness_population = _FITNESS_FUNCTIONS[_current_fitness_type]
        sample_environments = _NOISE_TYPES[_current_noise_type]
    else:
        raise ValueError(f"Unknown function type: {setting}")


# =============================================================================
# EXPERIMENT LOGGER
# =============================================================================

class ExperimentLogger:
    """
    Handles logging and saving of experiment data.

    Parameters
    ----------
    exp_name : str
        Experiment name (used for saving)
    exp_parameters : dict
        Experiment parameters
    exp_log_items : dict, optional
        Items to log with their shapes and dtypes
    """

    def __init__(self, exp_name, exp_parameters, exp_log_items=None):
        self.exp_name = exp_name
        self.exp_parameters = exp_parameters
        self.data = {}
        T = exp_parameters["T"]

        if exp_log_items is not None:
            for item in exp_log_items:
                shape = exp_log_items[item]["shape"]
                dtype = exp_log_items[item]["dtype"]
                self.data[item] = np.zeros(shape=(T, *shape), dtype=dtype)

    def log_iteration(self, data, t):
        """Log data for iteration t."""
        for item in data:
            self.data[item][t] = data[item]

    def close(self, to_compute):
        """Compute derived quantities after simulation."""
        if "last_iter_class_freqs" in to_compute:
            ks = self.exp_parameters["ks"]
            class_size = self.exp_parameters.get("class_size",
                         self.exp_parameters.get("class_size_list", [0])[0])
            last_iter_class_freqs = np.sum(
                self.data["frequency"][-1].reshape((len(ks), class_size)),
                axis=1
            )
            self.data["last_iter_class_freqs"] = last_iter_class_freqs

    def save(self, to_save="all"):
        """Save experiment data to file."""
        if to_save == "all":
            save_obj = self
        else:
            save_obj = ExperimentLogger(self.exp_name, self.exp_parameters)
            for item in to_save:
                save_obj.data[item] = self.data[item]

        os.makedirs("experiments", exist_ok=True)
        with open(f"experiments/{self.exp_name}.pkl", "wb") as f:
            pickle.dump(save_obj, f)


# =============================================================================
# POPULATION SAMPLING
# =============================================================================

def sample_parameters_population(n, ks, class_size_list, gt=None):
    """
    Generate population of organisms across complexity classes.

    Parameters
    ----------
    n : int
        Dimension
    ks : list
        Complexity classes
    class_size_list : list
        Number of individuals per class
    gt : array, optional
        Ground truth to insert as last individual

    Returns
    -------
    array
        Population of organisms
    """
    shape = sample_parameters(n, ks[0]).shape
    population_M = np.zeros(shape=(sum(class_size_list), *shape))
    indices = np.concatenate([[0], np.cumsum(class_size_list)])

    for i, k in enumerate(ks):
        for j in range(indices[i], indices[i + 1]):
            population_M[j, :] = sample_parameters(n, k)

    if gt is not None:
        population_M[-1, :] = gt

    return population_M


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run(exp_name, true_k, fitness_gamma=0.05, to_save=None, n=3, T=1000,
        ks=None, class_size=2000, env_switches=None, n_envs=None,
        env_change_rate=None, xi=0.1, to_return=None, insert_gt=False,
        adaptive_classes=False, init_frequency=None, ar1_rho=None,
        cue_ar1_rho=None):
    """
    Run replicator dynamics simulation.

    Implements the discrete-time replicator equation:
        x_i^(t+1) ∝ x_i^(t) * f_i^(t)

    Parameters
    ----------
    exp_name : str
        Experiment name for saving
    true_k : int
        Environmental complexity (q*)
    fitness_gamma : float
        Selection strength (γ)
    to_save : list, optional
        Items to save to file
    n : int
        Matrix/vector dimension
    T : int
        Number of timesteps
    ks : list
        Complexity classes to simulate
    class_size : int or list
        Individuals per complexity class
    env_switches : list, optional
        Explicit timesteps when environment changes
    n_envs : int, optional
        Number of equal-length environment periods
    env_change_rate : float, optional
        Poisson rate for environment changes
    xi : float
        Noise level (σ in paper)
    to_return : list, optional
        Items to return: 'frequency', 'class_frequency', 'fitness'
    insert_gt : bool
        Whether to insert ground truth as last individual
    adaptive_classes : bool
        Use adaptive class dynamics (experimental)
    init_frequency : array, optional
        Initial frequency distribution
    ar1_rho : float, optional
        AR(1) autocorrelation for successive ground truth matrices.
        When environment changes, A_new = rho * A_old + sqrt(1-rho^2) * innovation.
        rho=0: independent (same as default). rho->1: tiny perturbations.
        If None, successive environments are independent (original behavior).
    cue_ar1_rho : float, optional
        AR(1) autocorrelation for input signals (cues).
        ξ(t+1) = rho * ξ(t) + sqrt(1-rho^2) * ε. Stationary with marginal N(0,I).
        If None, input signals are iid (original behavior).

    Returns
    -------
    dict or None
        Requested data items if to_return is specified
    """
    if ks is None:
        ks = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Handle environment switching
    if env_switches is None:
        if n_envs is not None:
            env_switches = [int(T / n_envs * i) for i in range(n_envs)] + [T]
        elif env_change_rate is not None:
            env_idxs = np.zeros(T, dtype=int)
            env_idxs[0] = 0
            env_switches = [0]
            for t in range(1, T):
                if np.random.rand() < env_change_rate:
                    env_switches.append(t)
            env_switches.append(T)
        else:
            env_switches = [0, T]  # Single environment

    # Handle class sizes
    if isinstance(class_size, list):
        class_size_list = class_size
    else:
        class_size_list = [class_size] * len(ks)

    if insert_gt:
        class_size_list = class_size_list + [1]

    n_types = sum(class_size_list)

    # Determine what data to log based on what's needed
    # Only allocate arrays for data we actually need (saves memory!)
    items_to_log = set()
    if to_return is not None:
        items_to_log.update(to_return)
    if to_save is not None:
        if to_save == "all":
            items_to_log.update(["frequency", "class_frequency", "fitness"])
        else:
            items_to_log.update(to_save)

    # Always need class_frequency for internal use
    items_to_log.add("class_frequency")

    exp_log_items = {}
    if "frequency" in items_to_log and not adaptive_classes:
        exp_log_items["frequency"] = {"shape": (n_types,), "dtype": float}
    if "class_frequency" in items_to_log:
        exp_log_items["class_frequency"] = {"shape": (len(ks) + (1 if insert_gt else 0),), "dtype": float}
    if "fitness" in items_to_log:
        exp_log_items["fitness"] = {"shape": (n_types,), "dtype": float}
    if "max_fitness_class" in items_to_log:
        # Compact: just store which class has max fitness each timestep
        exp_log_items["max_fitness_class"] = {"shape": (), "dtype": int}

    # Initialize logger
    exp_logger = ExperimentLogger(
        exp_name,
        exp_parameters={
            "T": T, "n": n, "ks": ks, "class_size_list": class_size_list,
            "n_types": n_types, "true_k": true_k,
            "fitness_gamma": fitness_gamma, "env_switches": env_switches,
            "env_change_rate": env_change_rate, "ar1_rho": ar1_rho,
            "cue_ar1_rho": cue_ar1_rho
        },
        exp_log_items=exp_log_items
    )

    # Generate environments
    E = np.zeros((T, 2, n))
    env_slices = [slice(env_switches[i], env_switches[i + 1])
                  for i in range(len(env_switches) - 1)]

    env_M = None
    for env_slice in env_slices:
        if env_M is None or ar1_rho is None:
            # First environment or no AR(1): draw independent A
            env_M = sample_parameters(n, true_k)
        else:
            # AR(1) update: A_new = rho * A_old + sqrt(1 - rho^2) * innovation
            innovation = sample_parameters(n, true_k)
            env_M = ar1_rho * env_M + np.sqrt(1 - ar1_rho**2) * innovation
        E[env_slice, :, :] = sample_environments(
            env_slice.stop - env_slice.start, env_M, n, xi,
            cue_ar1_rho=cue_ar1_rho
        )

    # Generate population
    population_M = sample_parameters_population(n, ks, class_size_list) \
        if not insert_gt else \
        sample_parameters_population(n, ks, class_size_list, gt=env_M)

    # Initialize frequencies
    frequency = np.ones(n_types) / n_types if init_frequency is None \
        else init_frequency

    # Main evolution loop
    if not adaptive_classes:
        for t in range(T):
            fitness = fitness_population(population_M, E[t], fitness_gamma)
            class_freqs = chunk_sum_reduceat(frequency, class_size_list)

            # Only log what's needed
            log_data = {}
            if "frequency" in items_to_log:
                log_data["frequency"] = frequency.copy()
            if "class_frequency" in items_to_log:
                log_data["class_frequency"] = class_freqs
            if "fitness" in items_to_log:
                log_data["fitness"] = fitness
            if "max_fitness_class" in items_to_log:
                # Compute max fitness per class and return the class index
                max_fit_per_class = compute_max_fitness_per_class(fitness, class_size_list)
                log_data["max_fitness_class"] = np.argmax(max_fit_per_class)
            exp_logger.log_iteration(log_data, t=t)

            # Replicator equation
            frequency = frequency * fitness
            frequency = frequency / np.sum(frequency)

    # Save and return
    if to_save is not None:
        exp_logger.save(to_save=to_save)

    if to_return is not None:
        return {item: exp_logger.data[item] for item in to_return}


# =============================================================================
# MUTATION SIMULATION
# =============================================================================

def run_with_mutation(exp_name, true_k, initial_k, mutation_rate=None,
                      mutation_rate_between=0.01, mutation_rate_within=0.0,
                      M=None, fitness_gamma=0.05, n=3, T=1000,
                      class_size=500, xi=0.1, n_envs=1,
                      track_invasions=True, invasion_threshold=0.5,
                      min_k=1, max_k=9, to_return=None, seed=None):
    """
    Run replicator dynamics with mutation between and within complexity classes.

    This simulation allows two types of mutations:
    - Between-class (μ_b): promotes (adds parameter) or demotes (zeros parameter)
    - Within-class (μ_w): resamples all parameters, keeping same complexity

    Each timestep:
    - Poisson(M * μ_b) between-class mutations occur
    - Poisson(M * μ_w) within-class mutations occur
    - Mutants are added with frequency 1/M
    - Types with frequency < 1/M go extinct

    Parameters
    ----------
    exp_name : str
        Experiment name
    true_k : int
        Environmental complexity (q*)
    initial_k : int
        Starting complexity class (q^0)
    mutation_rate : float, optional
        Backward compatibility alias for mutation_rate_between.
        If provided, overrides mutation_rate_between.
    mutation_rate_between : float
        Per-individual per-timestep between-class mutation rate (μ_b).
        Default: 0.01
    mutation_rate_within : float
        Per-individual per-timestep within-class mutation rate (μ_w).
        Default: 0.0 (no within-class mutations)
    M : int, optional
        Number of individuals. Default: 100 * class_size
    fitness_gamma : float
        Selection strength (γ)
    n : int
        Matrix dimension
    T : int
        Number of timesteps
    class_size : int
        Initial number of types in starting class
    xi : float
        Environmental noise level
    n_envs : int
        Number of environment changes (1 = static environment)
    track_invasions : bool
        Whether to track successful invasion events
    invasion_threshold : float
        Frequency threshold for "takeover" (default 0.5)
    min_k : int
        Minimum complexity class (cannot demote below)
    max_k : int
        Maximum complexity class (cannot promote above)
    to_return : list, optional
        Items to return. Options: 'class_frequency', 'avg_complexity',
        'n_types', 'invasion_events', 'mutation_count',
        'mutation_count_between', 'mutation_count_within'
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Requested data items
    """
    if seed is not None:
        np.random.seed(seed)

    # Backward compatibility: mutation_rate overrides mutation_rate_between
    if mutation_rate is not None:
        mutation_rate_between = mutation_rate

    # Setup complexity classes
    ks = list(range(min_k, max_k + 1))

    # Default M if not specified
    if M is None:
        M = 100 * class_size

    # Initialize population in the initial complexity class
    population = MutationPopulation(n=n, M=M, ks=ks)
    population.initialize_from_class(initial_k, class_size)

    # Initialize lineage tracker
    lineage_tracker = LineageTracker(invasion_threshold=invasion_threshold) \
        if track_invasions else None

    # Setup environments
    env_switches = [int(T / n_envs * i) for i in range(n_envs)] + [T]
    E = np.zeros((T, 2, n))

    for i in range(len(env_switches) - 1):
        env_slice = slice(env_switches[i], env_switches[i + 1])
        env_M = sample_parameters_linear(n, true_k)
        E[env_slice, :, :] = sample_environments_linear(
            env_slice.stop - env_slice.start, env_M, n, xi
        )

    # Storage for results
    class_frequency_history = np.zeros((T, len(ks)))
    avg_complexity_history = np.zeros(T)
    n_types_history = np.zeros(T, dtype=int)
    mutation_count_between = 0
    mutation_count_within = 0

    # Main evolution loop
    for t in range(T):
        # Get population as array for fitness computation
        pop_array = population.get_types_array()

        if len(pop_array) == 0:
            # Population went extinct - shouldn't happen but handle gracefully
            break

        # Calculate fitness
        fitness = fitness_population_linear(pop_array, E[t], fitness_gamma)

        # Log current state
        class_freqs = population.get_class_frequencies()
        class_frequency_history[t] = class_freqs
        avg_complexity_history[t] = compute_avg_complexity(class_freqs, ks)
        n_types_history[t] = len(population)

        # MUTATION STEP - Between-class mutations (promote/demote)
        n_between = np.random.poisson(M * mutation_rate_between)
        for _ in range(n_between):
            if len(population) == 0:
                break

            # Select parent proportional to frequency
            parent_idx = np.random.choice(len(population), p=population.frequencies)
            parent_k = population.complexities[parent_idx]
            parent_matrix = population.types[parent_idx]

            # Determine mutation direction (with boundary handling)
            if parent_k <= min_k:
                direction = 'promote'
            elif parent_k >= max_k:
                direction = 'demote'
            else:
                direction = 'promote' if np.random.rand() < 0.5 else 'demote'

            # Create mutant
            try:
                if direction == 'promote':
                    mutant_matrix = mutate_promote(parent_matrix, n, parent_k)
                    mutant_k = parent_k + 1
                else:
                    mutant_matrix = mutate_demote(parent_matrix, n, parent_k)
                    mutant_k = parent_k - 1

                # Add mutant to population
                type_id = population.add_mutant(mutant_matrix, mutant_k, birth_time=t)
                mutation_count_between += 1

                # Register with lineage tracker
                if track_invasions:
                    lineage_tracker.register_mutant(type_id, t, mutant_k, mutation_type='between')

            except ValueError:
                # Mutation not possible (at boundary)
                pass

        # MUTATION STEP - Within-class mutations (resample parameters)
        n_within = np.random.poisson(M * mutation_rate_within)
        for _ in range(n_within):
            if len(population) == 0:
                break

            # Select parent proportional to frequency
            parent_idx = np.random.choice(len(population), p=population.frequencies)
            parent_k = population.complexities[parent_idx]
            parent_matrix = population.types[parent_idx]

            # Create within-class mutant (same complexity, resampled parameters)
            mutant_matrix = mutate_within_class(parent_matrix, n, parent_k)
            mutant_k = parent_k  # complexity unchanged

            # Add mutant to population
            type_id = population.add_mutant(mutant_matrix, mutant_k, birth_time=t)
            mutation_count_within += 1

            # Register with lineage tracker
            if track_invasions:
                lineage_tracker.register_mutant(type_id, t, mutant_k, mutation_type='within')

        # Replicator dynamics (selection)
        if len(population) > 0:
            pop_array = population.get_types_array()
            fitness = fitness_population_linear(pop_array, E[t], fitness_gamma)
            population.frequencies = population.frequencies * fitness
            freq_sum = population.frequencies.sum()
            if freq_sum > 0:
                population.frequencies = population.frequencies / freq_sum

        # Remove extinct types
        extinct_ids = population.remove_extinct()

        # Update lineage tracker
        if track_invasions:
            lineage_tracker.update(t, population)

    # Prepare return values
    result = {}
    if to_return is None:
        to_return = ['class_frequency', 'avg_complexity', 'invasion_events']

    if 'class_frequency' in to_return:
        result['class_frequency'] = class_frequency_history
    if 'avg_complexity' in to_return:
        result['avg_complexity'] = avg_complexity_history
    if 'n_types' in to_return:
        result['n_types'] = n_types_history
    if 'mutation_count' in to_return:
        # Total mutations (backward compatibility)
        result['mutation_count'] = mutation_count_between + mutation_count_within
    if 'mutation_count_between' in to_return:
        result['mutation_count_between'] = mutation_count_between
    if 'mutation_count_within' in to_return:
        result['mutation_count_within'] = mutation_count_within
    if 'invasion_events' in to_return and track_invasions:
        result['invasion_events'] = lineage_tracker.get_invasion_events()
    if 'ks' in to_return:
        result['ks'] = ks

    # Always include ks for convenience
    result['ks'] = ks

    return result


# =============================================================================
# METRICS FOR PAPER FIGURES
# =============================================================================

def compute_class_fitness(frequency, fitness, class_size_list):
    """
    Compute class fitness F_k (Eq. 6 in paper).

    F_k = sum_i(x_i * f_i) / X_k

    Parameters
    ----------
    frequency : array (n_types,)
        Individual frequencies
    fitness : array (n_types,)
        Individual fitness values
    class_size_list : list
        Number of individuals per class

    Returns
    -------
    array (n_classes,)
        Class fitness values
    """
    class_fitness = []
    idx = 0
    for size in class_size_list:
        class_slice = slice(idx, idx + size)
        X_k = np.sum(frequency[class_slice])
        if X_k > 0:
            F_k = np.sum(frequency[class_slice] * fitness[class_slice]) / X_k
        else:
            F_k = 0
        class_fitness.append(F_k)
        idx += size
    return np.array(class_fitness)


def compute_max_fitness_per_class(fitness, class_size_list):
    """
    Compute maximum fitness in each complexity class.

    Parameters
    ----------
    fitness : array (n_types,)
        Individual fitness values
    class_size_list : list
        Number of individuals per class

    Returns
    -------
    array (n_classes,)
        Maximum fitness per class
    """
    max_fitness = []
    idx = 0
    for size in class_size_list:
        class_slice = slice(idx, idx + size)
        max_fitness.append(np.max(fitness[class_slice]))
        idx += size
    return np.array(max_fitness)


def compute_occam_factor(frequency_t, frequency_t1, fitness_t, class_size_list):
    """
    Compute Occam factor for each class (Eq. 5 in paper).

    Occam factor = x̃_i*(t) / x̃_i*(t+1) where i* is best member at time t.
    A smaller Occam factor indicates more "collapse" toward the current best.

    Parameters
    ----------
    frequency_t : array (n_types,)
        Frequencies at time t
    frequency_t1 : array (n_types,)
        Frequencies at time t+1
    fitness_t : array (n_types,)
        Fitness at time t
    class_size_list : list
        Number of individuals per class

    Returns
    -------
    array (n_classes,)
        Occam factors
    """
    occam_factors = []
    idx = 0
    for size in class_size_list:
        class_slice = slice(idx, idx + size)

        # Find best member at time t
        i_star = np.argmax(fitness_t[class_slice]) + idx

        # Within-class relative frequencies
        X_k_t = np.sum(frequency_t[class_slice])
        X_k_t1 = np.sum(frequency_t1[class_slice])

        if X_k_t > 0 and X_k_t1 > 0:
            x_tilde_t = frequency_t[i_star] / X_k_t
            x_tilde_t1 = frequency_t1[i_star] / X_k_t1
            if x_tilde_t1 > 0:
                occam_factors.append(x_tilde_t / x_tilde_t1)
            else:
                occam_factors.append(np.nan)
        else:
            occam_factors.append(np.nan)

        idx += size

    return np.array(occam_factors)


def compute_class_growth_rate(class_freq_t1, class_freq_t):
    """
    Compute normalized class growth rate.

    Parameters
    ----------
    class_freq_t1 : array (n_classes,)
        Class frequencies at time t+1
    class_freq_t : array (n_classes,)
        Class frequencies at time t

    Returns
    -------
    array (n_classes,)
        Growth rates (normalized)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        growth = class_freq_t1 / class_freq_t
        # Normalize to mean 1
        growth = growth / np.nanmean(growth)
    return growth


def compute_selected_complexity(class_freqs, ks):
    """
    Compute mean selected complexity from class frequencies.

    Parameters
    ----------
    class_freqs : array (n_classes,)
        Class frequencies (should sum to 1)
    ks : list
        Complexity values for each class

    Returns
    -------
    float
        Expected complexity (weighted average)
    """
    return np.sum(class_freqs * np.array(ks))


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def setup_plot_style():
    """Set up matplotlib style for paper figures."""
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        'figure.titlesize': 26,
        'figure.titleweight': 'bold',
        'axes.titlesize': 22,
        'axes.titleweight': 'normal',
        'axes.labelsize': 24,
        'axes.labelweight': 'normal',
        'ytick.labelsize': 20,
        'xtick.labelsize': 20,
        'legend.fontsize': 18,
        'figure.figsize': (8, 6),
        'savefig.dpi': 300
    })


def plot_muller(organism_freqs, organism_colors=None, ax=None,
                class_freqs=None, class_colors=None, gamma=1, ks=None):
    """
    Plot Muller diagram of population dynamics.

    Parameters
    ----------
    organism_freqs : array (T, population_size)
        Frequency matrix where each row sums to 1
    organism_colors : array, optional
        Colors for each organism
    ax : matplotlib axis, optional
        Axis to plot on
    class_freqs : array (T, n_classes), optional
        Class frequencies for overlay
    class_colors : array, optional
        Colors for each class
    gamma : float
        For scaling x-axis labels (generations = t * gamma)
    ks : list, optional
        Complexity class labels for legend

    Returns
    -------
    ax : matplotlib axis
    """
    T, population_size = organism_freqs.shape
    assert np.allclose(organism_freqs.sum(axis=1), np.ones(T)), \
        "Each row must sum to 1"

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    y_stack = np.cumsum(organism_freqs, axis=1)

    if organism_colors is not None:
        ax.fill_between(range(T), 0, y_stack[:, 0], step='mid',
                       color=organism_colors[0])
        for i in range(1, population_size):
            ax.fill_between(range(T), y_stack[:, i-1], y_stack[:, i],
                           step='mid', color=organism_colors[i])
    else:
        ax.fill_between(range(T), 0, y_stack[:, 0], step='mid')
        for i in range(1, population_size):
            ax.fill_between(range(T), y_stack[:, i-1], y_stack[:, i], step='mid')

    # Plot class frequency boundaries
    if class_freqs is not None:
        ax.plot(np.cumsum(class_freqs[:, :-1], axis=1), ls="--",
               color='k', linewidth=2)

        # Legend at t=0
        if class_colors is None:
            n_classes = class_freqs.shape[1]
            class_colors = plt.cm.viridis(np.linspace(0, 1, n_classes))

        y_1, y_2 = 0, 0
        for type_i, type_color in enumerate(class_colors):
            y_2 += class_freqs[0, type_i]
            ax.fill_between(np.arange(-T//20, 0.01), y_1, y_2,
                           color=type_color, alpha=0.5)
            y_1 += class_freqs[0, type_i]

    if gamma != 1:
        labels = ax.get_xticks()
        ax.set_xticklabels([f"{l * gamma:.0f}" for l in labels])

    ax.set_xlabel('Generations')
    ax.set_ylabel('Frequency')
    ax.set_xlim(-T//20 if class_freqs is not None else 0, T)
    ax.set_ylim(0, 1)

    return ax


def plot_bubble_chart(M, ax=None, xlabel=None, ylabel=None):
    """
    Plot bubble chart for complexity selection.

    Parameters
    ----------
    M : array (n_q_stars, n_bins)
        Selection probability matrix
    ax : matplotlib axis, optional
    xlabel : str, optional
    ylabel : str, optional

    Returns
    -------
    ax : matplotlib axis
    """
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plt.rcParams.update({
        'axes.labelsize': 28,
        'ytick.labelsize': 28,
        'xtick.labelsize': 28
    })

    n_q_stars, n_bins = M.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(n_q_stars):
        arg_max_i = np.argmax(M[i])
        for j in range(n_bins):
            bubble_size = M[i, j] * 1000
            color = 'steelblue' if j == arg_max_i else 'lightgray'
            ax.scatter(i + 1, j + 1, s=bubble_size, color=color,
                      alpha=0.6, edgecolors='black')

    ax.set_xticks(np.arange(1, n_q_stars + 1))
    ax.set_yticks(np.arange(1, n_bins + 1))

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    return ax


def plot_env_change_heatmap(mean_selected_class, true_ks, n_envs_list,
                            ax=None, cmap='inferno', vmin=None, vmax=None):
    """
    Plot heatmap of selected complexity vs (q*, environmental change rate).

    For Figures 4E, 5B, 6B.

    Parameters
    ----------
    mean_selected_class : array (len(n_envs_list), len(true_ks))
        Mean selected complexity for each condition
    true_ks : list
        Environmental complexity values (x-axis)
    n_envs_list : list
        Environmental change rates (y-axis)
    ax : matplotlib axis, optional
    cmap : str
        Colormap
    vmin, vmax : float, optional
        Color limits

    Returns
    -------
    ax, im : matplotlib axis and image
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if vmin is None:
        vmin = true_ks[0]
    if vmax is None:
        vmax = true_ks[-1]

    im = ax.imshow(mean_selected_class, cmap=cmap, origin='lower',
                   vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xlabel(r"Environmental Complexity ($q^*$)")
    ax.set_ylabel("Environmental change rate")
    ax.set_xticks(np.arange(len(true_ks)))
    ax.set_xticklabels(true_ks)
    ax.set_yticks(np.arange(len(n_envs_list)))
    ax.set_yticklabels(n_envs_list)

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label(r"Selected complexity ($\langle q_\infty \rangle$)")

    return ax, im


def plot_invasion_heatmap(invasion_success, ks, ax=None, title=None):
    """
    Plot invasion success heatmap.

    Parameters
    ----------
    invasion_success : array (n_ks, n_ks)
        Invasion success matrix (entry [i,j] = success of k_i invading k_j)
    ks : list
        Complexity classes
    ax : matplotlib axis, optional
    title : str, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Mask diagonal
    masked_data = invasion_success.copy()
    np.fill_diagonal(masked_data, np.nan)

    im = ax.imshow(masked_data, cmap='Blues', vmin=0, vmax=1,
                   origin='lower', aspect='equal')

    # Gray diagonal
    for i in range(len(ks)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   facecolor='lightgray', edgecolor='black'))

    ax.set_xlabel(r'Resident $q$')
    ax.set_ylabel(r'Invading $q$')
    ax.set_xticks(range(len(ks)))
    ax.set_yticks(range(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_yticklabels(ks)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Invasion Success')

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax


def plot_dynamics_panel(data, ks, ax=None, xlabel='q', ylabel='Value',
                        colors=None, show_errorbars=True):
    """
    Plot dynamics panel with error bars (for Fig 2-4 C-F panels).

    Parameters
    ----------
    data : array (n_realizations, n_classes) or (n_classes,)
        Data to plot
    ks : list
        Complexity classes (x-axis)
    ax : matplotlib axis, optional
    xlabel, ylabel : str
    colors : array, optional
    show_errorbars : bool

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if data.ndim == 1:
        means = data
        stds = None
    else:
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0) / np.sqrt(data.shape[0])  # SEM

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))

    if show_errorbars and stds is not None:
        ax.errorbar(ks, means, yerr=1.96 * stds, fmt='o-', capsize=3)
    else:
        ax.plot(ks, means, 'o-')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ks)

    return ax


def plot_fitness_distribution(fitness_by_class, ks, ax=None):
    """
    Plot fitness distributions for different complexity classes (Fig 3B).

    Parameters
    ----------
    fitness_by_class : list of arrays
        Fitness values for each class
    ks : list
        Complexity classes
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))

    for i, (fitness, k) in enumerate(zip(fitness_by_class, ks)):
        # Plot histogram
        ax.hist(fitness, bins=30, alpha=0.5, color=colors[i],
                label=f'q={k}', density=True)
        # Plot mean
        ax.axvline(np.mean(fitness), color=colors[i], linestyle='--', linewidth=2)

    ax.set_xlabel('Fitness')
    ax.set_ylabel('Frequency')
    ax.legend()

    return ax


# =============================================================================
# MUTATION VISUALIZATION FUNCTIONS
# =============================================================================

def plot_complexity_trajectory(avg_complexity, T=None, true_k=None, initial_k=None,
                               ax=None, label=None, color=None):
    """
    Plot average complexity over time with reference lines.

    Parameters
    ----------
    avg_complexity : array (T,)
        Average complexity at each timestep
    T : int, optional
        Number of timesteps (inferred from data if not provided)
    true_k : int, optional
        Environmental complexity (target) - draws horizontal line
    initial_k : int, optional
        Starting complexity - draws horizontal line
    ax : matplotlib axis, optional
        Axis to plot on
    label : str, optional
        Label for the trajectory line
    color : str, optional
        Color for the trajectory line

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if T is None:
        T = len(avg_complexity)

    # Plot trajectory
    line_kwargs = {'linewidth': 2}
    if label is not None:
        line_kwargs['label'] = label
    if color is not None:
        line_kwargs['color'] = color

    ax.plot(range(T), avg_complexity[:T], **line_kwargs)

    # Reference lines
    if true_k is not None:
        ax.axhline(y=true_k, color='green', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Target q* = {true_k}')
    if initial_k is not None:
        ax.axhline(y=initial_k, color='red', linestyle=':', linewidth=1.5,
                   alpha=0.7, label=f'Initial q⁰ = {initial_k}')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Complexity')
    ax.set_title('Evolution of Population Complexity')
    ax.legend(loc='best')
    ax.set_ylim(0, 10)

    return ax


def plot_mutation_muller(class_freqs, ks, invasion_events=None, ax=None,
                         true_k=None, initial_k=None, cmap='viridis'):
    """
    Muller diagram for mutation simulations with optional invasion markers.

    Parameters
    ----------
    class_freqs : array (T, n_classes)
        Class frequencies over time
    ks : list
        Complexity classes
    invasion_events : list of dict, optional
        Invasion events from LineageTracker
    ax : matplotlib axis, optional
    true_k : int, optional
        Environmental complexity for title
    initial_k : int, optional
        Initial complexity for title
    cmap : str
        Colormap name

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    T, n_classes = class_freqs.shape
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_classes))

    # Create stacked area plot
    y_stack = np.cumsum(class_freqs, axis=1)

    # Fill areas from bottom to top
    ax.fill_between(range(T), 0, y_stack[:, 0], color=colors[0],
                    label=f'q={ks[0]}', alpha=0.8)
    for i in range(1, n_classes):
        ax.fill_between(range(T), y_stack[:, i-1], y_stack[:, i],
                        color=colors[i], label=f'q={ks[i]}', alpha=0.8)

    # Mark invasion events
    if invasion_events is not None and len(invasion_events) > 0:
        for event in invasion_events:
            birth_t = event['birth_time']
            takeover_t = event['takeover_time']
            birth_class = event['birth_class']

            # Vertical lines for invasion
            ax.axvline(x=birth_t, color='white', linestyle=':', alpha=0.5, linewidth=1)
            ax.axvline(x=takeover_t, color='white', linestyle='-', alpha=0.8, linewidth=2)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)

    # Title
    title = 'Population Dynamics'
    if true_k is not None and initial_k is not None:
        title = f'Population Dynamics (q⁰={initial_k} → q*={true_k})'
    elif true_k is not None:
        title = f'Population Dynamics (q*={true_k})'
    ax.set_title(title)

    # Legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout()
    return ax


def plot_invasion_timeline(invasion_events, T, ks, ax=None):
    """
    Plot timeline of successful invasions by complexity class.

    Parameters
    ----------
    invasion_events : list of dict
        Invasion events from LineageTracker
    T : int
        Total simulation time
    ks : list
        Complexity classes
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    if not invasion_events:
        ax.text(0.5, 0.5, 'No successful invasions', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, T)
        ax.set_ylim(0, len(ks))
        return ax

    colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))

    for event in invasion_events:
        birth_t = event['birth_time']
        takeover_t = event['takeover_time']
        birth_class = event['birth_class']
        k_idx = ks.index(birth_class)

        # Draw arrow from birth to takeover
        ax.annotate('', xy=(takeover_t, k_idx), xytext=(birth_t, k_idx),
                    arrowprops=dict(arrowstyle='->', color=colors[k_idx],
                                   lw=2, mutation_scale=15))

        # Mark birth time
        ax.scatter(birth_t, k_idx, color=colors[k_idx], s=50, zorder=5,
                   edgecolors='black', linewidths=1)

        # Mark takeover time
        ax.scatter(takeover_t, k_idx, color=colors[k_idx], s=100, zorder=5,
                   marker='*', edgecolors='black', linewidths=1)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Complexity Class')
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, len(ks) - 0.5)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks)
    ax.set_title('Invasion Timeline (● birth, ★ takeover)')

    return ax


# =============================================================================
# NEURAL NETWORK SUPPORT (optional)
# =============================================================================

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True

    class Net(nn.Module):
        """
        1-hidden-layer neural network for NN experiments.

        Parameters
        ----------
        input_size : int
            Input dimension
        hidden_size : int
            Number of hidden units (complexity)
        output_size : int
            Output dimension
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def create_nn_population(n_input, n_output, ks, class_size):
        """
        Create population of neural networks.

        Parameters
        ----------
        n_input : int
            Input dimension
        n_output : int
            Output dimension
        ks : list
            Hidden sizes (complexity classes)
        class_size : int
            Networks per class

        Returns
        -------
        list of lists
            Population organized by class
        """
        population = []
        for k in ks:
            class_pop = []
            for _ in range(class_size):
                net = Net(n_input, k, n_output)
                class_pop.append(net)
            population.append(class_pop)
        return population

    def fitness_nn(net, input_data, target, gamma=1.0):
        """
        Compute fitness for a neural network.

        Parameters
        ----------
        net : Net
            Neural network
        input_data : torch.Tensor
            Input data
        target : torch.Tensor
            Target output
        gamma : float
            Selection strength

        Returns
        -------
        float
            Fitness value
        """
        with torch.no_grad():
            output = net(input_data)
            error = torch.norm(output - target).item()
        return np.exp(-gamma * error)

except ImportError:
    TORCH_AVAILABLE = False
    Net = None
    create_nn_population = None
    fitness_nn = None


# =============================================================================
# CONVENIENCE FUNCTIONS FOR FIGURE GENERATION
# =============================================================================

def run_figure_experiment(true_ks, n_envs_list, n_realizations, T, class_size,
                          fitness_gamma, xi, function_type='linear',
                          save_path=None, verbose=True):
    """
    Run experiments for generating heatmap figures (4E, 5B, 6B).

    Parameters
    ----------
    true_ks : list
        Environmental complexity values
    n_envs_list : list
        Environmental change rates
    n_realizations : int
        Number of realizations per condition
    T : int
        Simulation length
    class_size : int
        Individuals per class
    fitness_gamma : float
        Selection strength
    xi : float
        Noise level
    function_type : str
        'linear' or 'polynomial'
    save_path : str, optional
        Path to save results
    verbose : bool
        Print progress

    Returns
    -------
    array (len(n_envs_list), len(true_ks), n_realizations, T, len(true_ks))
        All class frequencies
    """
    set_function_type(function_type)
    ks = list(range(1, 10))

    all_class_freqs = np.zeros((len(n_envs_list), len(true_ks),
                                n_realizations, T, len(ks)))

    total = len(n_envs_list) * len(true_ks) * n_realizations
    count = 0

    for n_envs_i, n_envs in enumerate(n_envs_list):
        for true_k_i, true_k in enumerate(true_ks):
            for r in range(n_realizations):
                result = run(
                    exp_name="__temp__",
                    true_k=true_k,
                    T=T,
                    ks=ks,
                    class_size=class_size,
                    n_envs=n_envs,
                    fitness_gamma=fitness_gamma,
                    xi=xi,
                    to_save=None,
                    to_return=["class_frequency"]
                )
                all_class_freqs[n_envs_i, true_k_i, r] = result["class_frequency"]

                count += 1
                if verbose and count % 100 == 0:
                    print(f"Progress: {count}/{total}")

    if save_path:
        np.save(save_path, all_class_freqs)

    return all_class_freqs


def load_or_run_experiment(data_path, run_func, *args, force_rerun=False, **kwargs):
    """
    Load existing data or run experiment if not available.

    Parameters
    ----------
    data_path : str
        Path to data file
    run_func : callable
        Function to run if data doesn't exist
    force_rerun : bool
        Force re-running even if data exists
    *args, **kwargs
        Arguments to pass to run_func

    Returns
    -------
    Data from file or freshly computed
    """
    if os.path.exists(data_path) and not force_rerun:
        print(f"Loading existing data from {data_path}")
        return np.load(data_path, allow_pickle=True)
    else:
        print(f"Running experiment (will save to {data_path})")
        result = run_func(*args, save_path=data_path, **kwargs)
        return result
