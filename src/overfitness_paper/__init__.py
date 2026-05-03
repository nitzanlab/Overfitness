"""
overfitness_paper: code accompanying Rappeport & Nitzan,
"Fitness and Overfitness: Implicit Regularization in Evolutionary Dynamics".

The bulk of the simulation engine (replicator dynamics, function classes,
fitness/noise variants, mutation, metrics, plotting) lives in `simulation`.
Invasion experiments live in `invasion`. Neural-network model code lives
in `models.neural_net`. Canonical default parameters live in `config`.
"""

from . import config
from . import simulation
from . import invasion
from . import metrics_helpers
from .metrics_helpers import run_with_metrics
from .simulation import (
    run,
    run_with_mutation,
    ExperimentLogger,
    MutationPopulation,
    LineageTracker,
    np_temp_seed,
    sample_parameters_linear,
    fitness_population_linear,
    sample_environments_linear,
    set_function_type,
    set_fitness_type,
    set_noise_type,
    compute_class_fitness,
    compute_max_fitness_per_class,
    compute_occam_factor,
    compute_class_growth_rate,
    compute_selected_complexity,
    compute_avg_complexity,
    plot_muller,
    plot_bubble_chart,
    plot_env_change_heatmap,
    plot_dynamics_panel,
    plot_fitness_distribution,
    plot_invasion_heatmap,
    plot_complexity_trajectory,
    plot_mutation_muller,
    plot_invasion_timeline,
    setup_plot_style,
    run_figure_experiment,
    load_or_run_experiment,
)
