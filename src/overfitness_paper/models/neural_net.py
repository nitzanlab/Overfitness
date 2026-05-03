"""
Two-layer ReLU neural-network model used for SI figures S2, S4, S5D-F, S6D-F.

Extracted (verbatim, then refactored) from run_nn_correlated.py.
"""

import numpy as np


def generate_ar1_cues(T, n, rho):
    """
    Generate T input signals of dimension n.
    rho == 0: iid N(0, I).
    rho  > 0: AR(1) process xi(t+1) = rho*xi(t) + sqrt(1-rho^2)*eps, eps~N(0,I).
    Stationary marginal is N(0, I).
    """
    if rho > 0:
        signals = np.zeros((T, n))
        sigma = np.sqrt(1 - rho ** 2)
        signals[0] = np.random.randn(n)
        for t in range(1, T):
            signals[t] = rho * signals[t - 1] + sigma * np.random.randn(n)
        return signals
    return np.random.randn(T, n)


def make_env_schedule(T, change_rate):
    """Integer index of which environment is active at each timestep."""
    env_indices = np.zeros(T, dtype=int)
    current_env = 0
    for t in range(1, T):
        if change_rate > 0 and np.random.rand() < change_rate:
            current_env += 1
        env_indices[t] = current_env
    return env_indices, current_env + 1


def sample_environment_nns(n_envs, true_k, n_in, n_out):
    """Sample ground-truth NN parameters for n_envs distinct environments."""
    return dict(
        W1=np.random.randn(n_envs, true_k, n_in),
        b1=np.random.randn(n_envs, true_k),
        W2=np.random.randn(n_envs, n_out, true_k),
        b2=np.random.randn(n_envs, n_out),
    )


def sample_population_nns(ks, class_size, n_in, n_out):
    """Sample population NN parameters, one batched array per complexity class."""
    return dict(
        W1=[np.random.randn(class_size, k, n_in) for k in ks],
        b1=[np.random.randn(class_size, k) for k in ks],
        W2=[np.random.randn(class_size, n_out, k) for k in ks],
        b2=[np.random.randn(class_size, n_out) for k in ks],
    )


def run_single(true_k, ks, class_size, T, fitness_gamma, xi, n_in, n_out,
               change_rate, cue_rho):
    """
    Run one realization of NN evolution.
    Returns class_freqs (T, n_ks): frequency of each complexity class over time.
    """
    n_ks = len(ks)
    n_types = n_ks * class_size

    env_indices, n_envs = make_env_schedule(T, change_rate)
    env = sample_environment_nns(n_envs, true_k, n_in, n_out)
    pop = sample_population_nns(ks, class_size, n_in, n_out)
    cues = generate_ar1_cues(T, n_in, cue_rho)

    freq = np.ones(n_types) / n_types
    class_freqs = np.zeros((T, n_ks))

    for t in range(T):
        signal = cues[t]
        e = env_indices[t]
        h_env = np.maximum(0, env['W1'][e] @ signal + env['b1'][e])
        opt = env['W2'][e] @ h_env + env['b2'][e] + xi

        for i in range(n_ks):
            class_freqs[t, i] = np.sum(freq[i * class_size:(i + 1) * class_size])

        all_fitness = np.empty(n_types)
        offset = 0
        for i, k in enumerate(ks):
            h = np.maximum(0, np.einsum('ckn,n->ck', pop['W1'][i], signal) + pop['b1'][i])
            y = np.einsum('cok,ck->co', pop['W2'][i], h) + pop['b2'][i]
            error = np.linalg.norm(y - opt, axis=1)
            all_fitness[offset:offset + class_size] = np.exp(-fitness_gamma * error)
            offset += class_size

        freq = freq * all_fitness
        freq /= freq.sum()

    return class_freqs


def run_experiment(true_ks, ks, class_size, T, fitness_gamma, xi, n_in, n_out,
                   change_rate, cue_rho, n_realizations, verbose=True):
    """Run all (true_k, realization) pairs for one condition.
    Returns (n_true_ks, n_realizations, T, n_ks)."""
    out = np.zeros((len(true_ks), n_realizations, T, len(ks)))
    total = len(true_ks) * n_realizations
    count = 0
    for tk_i, true_k in enumerate(true_ks):
        for r in range(n_realizations):
            out[tk_i, r] = run_single(true_k, ks, class_size, T, fitness_gamma,
                                      xi, n_in, n_out, change_rate, cue_rho)
            count += 1
            if verbose and count % 10 == 0:
                print(f'  [{count}/{total}] true_k={true_k}, r={r}')
    return out


def compute_bubble_matrix(data, ks):
    """
    data: (n_true_ks, n_realizations, T, n_ks).
    Returns M (n_ks, n_ks): M[i, j] = P(selected class j | true class i).
    """
    final = data[:, :, -1, :]
    selected = np.argmax(final, axis=2)
    n_ks = len(ks)
    M = np.zeros((n_ks, n_ks))
    for i in range(n_ks):
        for j in range(n_ks):
            M[i, j] = np.mean(selected[i] == j)
    return M
