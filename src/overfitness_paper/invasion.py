
def compute_fitness_tensor(m, n, l, matrix_size=3, fitness_gamma=0.05, xi=1):
    """
    Computes a fitness tensor for different complexity levels and environment matrices.

    Parameters:
    -----------
    m : int
        Number of environment matrices to generate for each true_k
    n : int
        Population size for each complexity level
    l : int
        Number of input environments to average fitness over
    matrix_size : int, default=3
        Size of the square matrices (n x n)
    fitness_gamma : float, default=1.0
        Selection strength parameter for fitness calculation
    xi : float, default=0.1
        Noise level in environment generation
    save_path : str, default="fitness_tensor.npy"
        Path to save the resulting tensor

    Returns:
    --------
    fitness_tensor : numpy.ndarray
        Tensor of shape (9, m, 9, n) containing average log fitness values
    """

    # Set to use linear functions
    set_function_type("linear")

    # Initialize the output tensor
    fitness_tensor = np.zeros((9, m, 9, n))

    # Generate all environment matrices first
    print("Generating environment matrices...")
    env_matrices = np.zeros((9, m, matrix_size, matrix_size))

    for true_k_idx, true_k in enumerate(range(1, 10)):  # true_k from 1 to 9
        for m_idx in range(m):
            env_matrices[true_k_idx, m_idx] = sample_parameters_non_empty_entries(matrix_size, true_k)

    # Generate all population matrices for each complexity k
    print("Generating population matrices...")
    population_matrices = np.zeros((9, n, matrix_size, matrix_size))

    for k_idx, k in enumerate(range(1, 10)):  # k from 1 to 9
        for ind_idx in range(n):
            population_matrices[k_idx, ind_idx] = sample_parameters_non_empty_entries(matrix_size, k)

    # Pre-generate all input environments for vectorized computation
    print("Pre-generating input environments...")
    input_signals = np.random.randn(l, matrix_size)

    print("Computing fitness tensor...")

    # Main computation loop
    for true_k_idx in range(9):
        print(f"Processing true_k = {true_k_idx + 1}/9")

        for m_idx in range(m):
            # Current environment matrix
            env_matrix = env_matrices[true_k_idx, m_idx]

            # Generate target responses for all l input environments
            # Shape: (l, matrix_size)
            target_responses = (env_matrix @ input_signals.T).T + xi * np.random.randn(l, matrix_size)

            # For each population complexity k
            for k_idx in range(9):
                # Current population: shape (n, matrix_size, matrix_size)
                current_population = population_matrices[k_idx]

                # Compute responses for all individuals and all environments
                # population_responses shape: (n, l, matrix_size)
                population_responses = np.einsum('nij,lj->nli', current_population, input_signals)

                # Compute fitness for all individuals across all l environments
                # Broadcast target_responses to shape (1, l, matrix_size) for broadcasting
                target_broadcast = target_responses[np.newaxis, :, :]  # shape: (1, l, matrix_size)

                # Compute squared differences: shape (n, l, matrix_size)
                diff_squared = (population_responses - target_broadcast) ** 2

                # Sum over matrix_size dimension to get squared norms: shape (n, l)
                squared_norms = np.sum(diff_squared, axis=2)

                # Compute log fitness (negative log of exponential): shape (n, l)
                log_fitness = -fitness_gamma * np.sqrt(squared_norms)

                # Average over l environments: shape (n,)
                avg_log_fitness = np.mean(log_fitness, axis=1)

                # Store in tensor
                fitness_tensor[true_k_idx, m_idx, k_idx, :] = avg_log_fitness

    # Save the tensor

    print("Computation complete!")
    print(f"Final tensor shape: {fitness_tensor.shape}")

    return fitness_tensor


def compute_competition_probabilities(fitness_tensor):
    """
    Computes the probability that a random member of k_test will outperform
    a random member of k_true on a random environment matrix sampled from k_true.

    Computes probabilities per environment matrix (comparing n vs n individuals)
    and then averages over all m environment matrices.

    Parameters:
    -----------
    fitness_tensor : numpy.ndarray
        Fitness tensor of shape (9, m, 9, n) from compute_fitness_tensor()
        - dim 0: true_k values [1-9]
        - dim 1: m environment matrices per true_k
        - dim 2: test_k values [1-9]
        - dim 3: n individuals per population

    Returns:
    --------
    prob_matrix : numpy.ndarray
        Matrix of shape (9, 9) where prob_matrix[i,j] is the probability that
        a random individual from complexity k_test=j+1 outperforms a random
        individual from complexity k_true=i+1 on a random environment matrix
        sampled from k_true=i+1
    """

    # Get dimensions
    num_k_true, m, num_k_test, n = fitness_tensor.shape
    assert num_k_true == 9 and num_k_test == 9, "Expected 9 complexity levels"

    # Initialize probability matrix
    prob_matrix = np.zeros((9, 9))

    print("Computing competition probabilities...")

    for k_true_idx in range(9):  # k_true from 1 to 9
        print(f"Processing k_true = {k_true_idx + 1}/9")

        for k_test_idx in range(9):  # k_test from 1 to 9

            # Store probabilities for each environment matrix
            env_probabilities = np.zeros(m)

            for env_idx in range(m):
                # Get fitness values for this specific environment matrix
                # k_true individuals: shape (n,)
                k_true_fitness = fitness_tensor[k_true_idx, env_idx, k_true_idx, :]
                # k_test individuals: shape (n,)
                k_test_fitness = fitness_tensor[k_true_idx, env_idx, k_test_idx, :]

                # Compute probability that random k_test > random k_true
                # for this specific environment matrix
                # Use broadcasting to compare all n x n pairs
                # k_test_fitness[:, None] has shape (n, 1)
                # k_true_fitness[None, :] has shape (1, n)
                # comparison_matrix has shape (n, n)
                comparison_matrix = k_test_fitness[:, None] > k_true_fitness[None, :]

                # Probability for this environment is fraction of True values
                env_probabilities[env_idx] = np.mean(comparison_matrix)

            # Average probability across all m environment matrices
            prob_matrix[k_true_idx, k_test_idx] = np.mean(env_probabilities)

    print("Competition probabilities computed!")
    return prob_matrix


def compute_competition_probabilities_old(fitness_tensor):
    """
    Computes the probability that a random member of k_test will outperform
    a random member of k_true on a random environment matrix sampled from k_true.

    Parameters:
    -----------
    fitness_tensor : numpy.ndarray
        Fitness tensor of shape (9, m, 9, n) from compute_fitness_tensor()
        - dim 0: true_k values [1-9]
        - dim 1: m environment matrices per true_k
        - dim 2: test_k values [1-9]
        - dim 3: n individuals per population

    Returns:
    --------
    prob_matrix : numpy.ndarray
        Matrix of shape (9, 9) where prob_matrix[i,j] is the probability that
        a random individual from complexity k_test=j+1 outperforms a random
        individual from complexity k_true=i+1 on a random environment matrix
        sampled from k_true=i+1
    """

    # Get dimensions
    num_k_true, m, num_k_test, n = fitness_tensor.shape
    assert num_k_true == 9 and num_k_test == 9, "Expected 9 complexity levels"

    # Initialize probability matrix
    prob_matrix = np.zeros((9, 9))

    print("Computing competition probabilities...")

    for k_true_idx in range(9):  # k_true from 1 to 9
        print(f"Processing k_true = {k_true_idx + 1}/9")

        # Get fitness values for k_true individuals on their own environments
        # Shape: (m, n) -> flatten to (m*n,)
        k_true_fitness = fitness_tensor[k_true_idx, :, k_true_idx, :].flatten()

        for k_test_idx in range(9):  # k_test from 1 to 9

            # Get fitness values for k_test individuals on k_true environments
            # Shape: (m, n) -> flatten to (m*n,)
            k_test_fitness = fitness_tensor[k_true_idx, :, k_test_idx, :].flatten()

            # Compute probability that random k_test > random k_true
            # This is equivalent to computing the fraction of all pairwise comparisons
            # where k_test individual beats k_true individual

            # Vectorized approach: use broadcasting to compare all pairs
            # k_test_fitness[:, None] has shape (m*n, 1)
            # k_true_fitness[None, :] has shape (1, m*n)
            # comparison_matrix has shape (m*n, m*n)
            comparison_matrix = k_test_fitness[:, None] > k_true_fitness[None, :]

            # Probability is the fraction of True values
            prob_matrix[k_true_idx, k_test_idx] = np.mean(comparison_matrix)

    print("Competition probabilities computed!")
    return prob_matrix


def compute_competition_probabilities_mean(fitness_tensor):
    """
    Computes the probability that a random member of k_test will outperform
    the MEAN fitness of k_true on a random environment matrix sampled from k_true.

    Computes probabilities per environment matrix (comparing n individuals
    to the mean of n other individuals) and then averages over all m environment matrices.

    Parameters:
    -----------
    fitness_tensor : numpy.ndarray
        Fitness tensor of shape (9, m, 9, n) from compute_fitness_tensor()
        - dim 0: true_k values [1-9]
        - dim 1: m environment matrices per true_k
        - dim 2: test_k values [1-9]
        - dim 3: n individuals per population

    Returns:
    --------
    prob_matrix : numpy.ndarray
        Matrix of shape (9, 9) where prob_matrix[i,j] is the probability that
        a random individual from complexity k_test=j+1 outperforms the mean
        fitness of complexity k_true=i+1 on a random environment matrix
        sampled from k_true=i+1
    """

    # Get dimensions
    num_k_true, m, num_k_test, n = fitness_tensor.shape
    assert num_k_true == 9 and num_k_test == 9, "Expected 9 complexity levels"

    # Initialize probability matrix
    prob_matrix = np.zeros((9, 9))

    print("Computing competition probabilities (vs mean)...")

    for k_true_idx in range(9):  # k_true from 1 to 9
        print(f"Processing k_true = {k_true_idx + 1}/9")

        for k_test_idx in range(9):  # k_test from 1 to 9

            # Store probabilities for each environment matrix
            env_probabilities = np.zeros(m)

            for env_idx in range(m):
                # Get fitness values for this specific environment matrix
                # k_true individuals: shape (n,)
                k_true_fitness = fitness_tensor[k_true_idx, env_idx, k_true_idx, :]
                # k_test individuals: shape (n,)
                k_test_fitness = fitness_tensor[k_true_idx, env_idx, k_test_idx, :]

                # Compute mean fitness of k_true population
                k_true_mean_fitness = np.mean(k_true_fitness)

                # Compute fraction of k_test individuals that beat the k_true mean
                env_probabilities[env_idx] = np.mean(k_test_fitness > k_true_mean_fitness)

            # Average probability across all m environment matrices
            prob_matrix[k_true_idx, k_test_idx] = np.mean(env_probabilities)

    print("Competition probabilities (vs mean) computed!")
    return prob_matrix



def analyze_competition_results(prob_matrix, save_path="competition_analysis.txt"):
    """
    Analyzes and saves the competition probability results.

    Parameters:
    -----------
    prob_matrix : numpy.ndarray
        Competition probability matrix from compute_competition_probabilities()
    save_path : str
        Path to save the analysis results
    """

    print("\n" + "=" * 60)
    print("COMPETITION PROBABILITY ANALYSIS")
    print("=" * 60)

    analysis = []
    analysis.append("Competition Probability Matrix")
    analysis.append("Rows: k_true (environment complexity)")
    analysis.append("Cols: k_test (population complexity)")
    analysis.append("Values: P(random k_test individual > random k_true individual)")
    analysis.append("")

    # Print matrix with headers
    header = "k_true\\k_test" + "".join([f"{k + 1:>8}" for k in range(9)])
    analysis.append(header)
    analysis.append("-" * len(header))

    for i in range(9):
        row = f"{i + 1:>11}" + "".join([f"{prob_matrix[i, j]:>8.3f}" for j in range(9)])
        analysis.append(row)
        print(f"k_true={i + 1}: " + " ".join([f"{prob_matrix[i, j]:.3f}" for j in range(9)]))

    analysis.append("")

    # Diagonal analysis (same complexity competition)
    analysis.append("Diagonal Elements (same complexity vs same complexity):")
    diagonal_values = np.diag(prob_matrix)
    for i, val in enumerate(diagonal_values):
        analysis.append(f"k={i + 1}: P(random k individual > another random k individual) = {val:.4f}")
    analysis.append(f"Mean diagonal value: {np.mean(diagonal_values):.4f} (should be ~0.5)")
    analysis.append("")

    # Above/below diagonal analysis
    upper_triangle = prob_matrix[np.triu_indices(9, k=1)]
    lower_triangle = prob_matrix[np.tril_indices(9, k=-1)]

    analysis.append("Upper Triangle (k_test > k_true):")
    analysis.append(f"Mean probability: {np.mean(upper_triangle):.4f}")
    analysis.append(f"Values > 0.5: {np.sum(upper_triangle > 0.5)} / {len(upper_triangle)}")
    analysis.append("")

    analysis.append("Lower Triangle (k_test < k_true):")
    analysis.append(f"Mean probability: {np.mean(lower_triangle):.4f}")
    analysis.append(f"Values > 0.5: {np.sum(lower_triangle > 0.5)} / {len(lower_triangle)}")
    analysis.append("")

    # Save analysis
    with open(save_path, 'w') as f:
        f.write('\n'.join(analysis))

    print(f"\nAnalysis saved to {save_path}")
    return prob_matrix



def plot_competition_heatmap(prob_matrix, figsize=(10, 8), cmap='RdYlBu_r',
                             save_path=None, show_values=True, title=None):
    """
    Creates a heatmap visualization of the competition probability matrix.

    Parameters:
    -----------
    prob_matrix : numpy.ndarray
        Competition probability matrix of shape (9, 9)
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches
    cmap : str, default='RdYlBu_r'
        Colormap for the heatmap. Options include:
        - 'RdYlBu_r': Red-Yellow-Blue (reversed)
        - 'viridis': Purple-Blue-Green-Yellow
        - 'plasma': Purple-Pink-Yellow
        - 'coolwarm': Blue-White-Red
    save_path : str, optional
        Path to save the figure (e.g., 'competition_heatmap.png')
    show_values : bool, default=True
        Whether to display probability values in each cell
    title : str, optional
        Custom title for the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    # Set up the plot style
    plt.style.use('default')  # Reset to default style
    sns.set_context("talk", font_scale=1.0)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Create the heatmap
    im = ax.imshow(prob_matrix, cmap=cmap, aspect='equal', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels([f'{k}' for k in range(1, 10)], fontsize=12)
    ax.set_yticklabels([f'{k}' for k in range(1, 10)], fontsize=12)

    # Labels and title
    ax.set_xlabel('Population Complexity ($k_{test}$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Environment Complexity ($k_{true}$)', fontsize=14, fontweight='bold')

    if title is None:
        title = 'Competition Probability Matrix\nP(random $k_{test}$ individual > random $k_{true}$ individual)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Probability', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # Add probability values to each cell
    if show_values:
        for i in range(9):
            for j in range(9):
                text_color = 'white' if prob_matrix[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{prob_matrix[i, j]:.3f}',
                        ha='center', va='center', fontsize=10,
                        color=text_color, fontweight='bold')

    # Add grid lines for better readability
    ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 9, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Heatmap saved to {save_path}")

    return fig, ax


def plot_competition_heatmap_advanced(prob_matrix, figsize=(12, 10), save_path=None):
    """
    Creates an advanced heatmap with additional annotations and styling.

    Parameters:
    -----------
    prob_matrix : numpy.ndarray
        Competition probability matrix of shape (9, 9)
    figsize : tuple, default=(12, 10)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    # Create figure with custom styling
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Create heatmap with seaborn for better default styling
    sns.heatmap(prob_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0.5,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Probability', 'shrink': 0.8},
                xticklabels=[f'k={k}' for k in range(1, 10)],
                yticklabels=[f'k={k}' for k in range(1, 10)],
                ax=ax)

    # Customize labels and title
    ax.set_xlabel('Population Complexity ($k_{test}$)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Environment Complexity ($k_{true}$)', fontsize=16, fontweight='bold')
    ax.set_title(
        'Competition Probability Matrix\nP(random $k_{test}$ individual outperforms random $k_{true}$ individual)',
        fontsize=18, fontweight='bold', pad=25)

    # Highlight diagonal
    for i in range(9):
        rect = plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', linewidth=3)
        ax.add_patch(rect)

    # Add text annotations for key regions
    ax.text(7.5, 1.5, 'Higher complexity\npopulations vs\nlower complexity\nenvironments',
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

    ax.text(1.5, 7.5, 'Lower complexity\npopulations vs\nhigher complexity\nenvironments',
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

    # Rotate y-axis labels for better readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Advanced heatmap saved to {save_path}")

    return fig, ax


def create_multiple_heatmap_views(prob_matrix, save_dir="./figures/"):
    """
    Creates multiple visualization views of the competition probability matrix.

    Parameters:
    -----------
    prob_matrix : numpy.ndarray
        Competition probability matrix of shape (9, 9)
    save_dir : str, default="./"
        Directory to save the plots

    Returns:
    --------
    dict of figure objects
    """

    figures = {}

    # 1. Standard heatmap
    print("Creating standard heatmap...")
    fig1, ax1 = plot_competition_heatmap(prob_matrix,
                                         save_path=f"{save_dir}/competition_heatmap_standard.png")
    figures['standard'] = (fig1, ax1)

    # 2. Advanced heatmap with annotations
    print("Creating advanced heatmap...")
    fig2, ax2 = plot_competition_heatmap_advanced(prob_matrix,
                                                  save_path=f"{save_dir}/competition_heatmap_advanced.png")
    figures['advanced'] = (fig2, ax2)

    # 3. Different colormap version
    print("Creating viridis colormap version...")
    fig3, ax3 = plot_competition_heatmap(prob_matrix, cmap='viridis',
                                         title='Competition Probabilities (Viridis Colormap)',
                                         save_path=f"{save_dir}/competition_heatmap_viridis.png")
    figures['viridis'] = (fig3, ax3)

    # 4. High contrast version without text
    print("Creating high contrast version...")
    fig4, ax4 = plot_competition_heatmap(prob_matrix, cmap='plasma', show_values=False,
                                         title='Competition Probabilities (High Contrast)',
                                         save_path=f"{save_dir}/competition_heatmap_contrast.png")
    figures['contrast'] = (fig4, ax4)

    print(f"All heatmaps saved to {save_dir}")
    return figures


# Example usage function
def visualize_competition_results(prob_matrix,
                                  save_visualizations=True):
    """
    Complete visualization pipeline for competition results.

    Parameters:
    -----------
    prob_matrix_path : str
        Path to the saved probability matrix
    save_visualizations : bool
        Whether to save the plots

    Returns:
    --------
    prob_matrix, figures
    """

    print(f"Got probability matrix of shape: {prob_matrix.shape}")

    # Create visualizations
    if save_visualizations:
        figures = create_multiple_heatmap_views(prob_matrix)
    else:
        fig, ax = plot_competition_heatmap(prob_matrix)
        figures = {'main': (fig, ax)}
        plt.show()

    return prob_matrix, figures

if __name__ == '__main__':
    m = 20
    n = 500
    l = 50
    matrix_size = 3
    fitness_gamma = 0.05
    xi = 1
    fitness_tensor = compute_fitness_tensor(m=m, n=n, l=l, matrix_size=matrix_size, fitness_gamma=fitness_gamma, xi=xi)
    save_path = f"fitness_tensor_{m=}_{n=}_{l=}_{fitness_gamma=}_{xi=}.npy"
    print(f"Saving fitness tensor to {save_path}")
    np.save(save_path, fitness_tensor)
    # fitness_tensor = np.load(save_path)
    prob_matrix = compute_competition_probabilities_mean(fitness_tensor)
    # save_path = f"competition_probs_{m=}_{n=}_{l=}_{fitness_gamma=}_{xi=}.npy"
    # prob_matrix_old = np.load(save_path)
    # np.save(save_path, prob_matrix)

    visualize_competition_results(prob_matrix)