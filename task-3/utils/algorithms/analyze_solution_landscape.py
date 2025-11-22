from typing import Any, Dict, List, Tuple, Optional, Iterable, Callable
from utils.initialization import initialize_random
from utils.plotting.solution_space_plots import (
    plot_similarity_metric_3d_landscapes_interactive,
    plot_similarity_metric_landscape,
)
import copy
from tqdm import tqdm
import random
import numpy as np
from utils.algorithms.local_search import greedy, steepest, generate_all_moves

MOVE_LIMIT = 10
POP_SIZE = 10
LS_SIZE = 2
NUM_OPTIMAL_OFFSPRINGS = 1
MAX_PERTURB_MOVES = 40


def analyze_solution_landscape(
    optimal: Any,
    analysis_dir: str,
    moves: Optional[Dict[str, Callable[[np.ndarray], Iterable[np.ndarray]]]] = None,
    metric_functions: Optional[Dict[str, Callable]] = None,
    optimization_directions: Optional[Dict[str, str]] = None,
) -> None:
    population: List[Any] = initialize_random(
        num_individuals=POP_SIZE,
        eager_metric_calculations=True,
        metric_functions=metric_functions,
    )

    similarities_random = calculate_all_similarities_metrics(optimal, population)
    ls_population = population[:LS_SIZE]
    all_similarities: Dict[
        str, Dict[str, Dict[str, List[Tuple[float, Dict[str, float]]]]]
    ] = {
        "Random": {},
        "Greedy": {},
        "Steepest": {},
        "Perturbed": {},
    }

    for metric_name in tqdm(metric_functions, desc="Analyzing Metrics"):
        all_similarities["Perturbed"][metric_name] = {}
        all_similarities["Random"][metric_name] = {}

        for move_name in moves:
            all_similarities["Random"][metric_name][move_name] = similarities_random

        for move_name, move in moves.items():
            direction = optimization_directions[metric_name]

            population_perturbed = generate_perturbed_population(
                optimal=optimal,
                move_func=move,
                num_offsprings=NUM_OPTIMAL_OFFSPRINGS,
                max_moves=MAX_PERTURB_MOVES,
            )

            sim_perturbed = calculate_all_similarities_metrics(
                optimal, population_perturbed
            )

            all_similarities["Perturbed"][metric_name][move_name] = copy.deepcopy(
                sim_perturbed
            )

    plot_similarity_metric_landscape(
        all_similarities=all_similarities,
        optimal=optimal,
        analysis_dir=analysis_dir,
        metric_functions=metric_functions,
        moves=moves,
    )

    plot_similarity_metric_3d_landscapes_interactive(
        all_similarities=all_similarities,
        optimal=optimal,
        analysis_dir=analysis_dir,
        metric_functions=metric_functions,
        moves=moves,
    )


def generate_perturbed_population(
    optimal: Any,
    move_func: Callable[[np.ndarray], Iterable[np.ndarray]],
    num_offsprings: int,
    max_moves: int,
) -> List[Any]:
    perturbed_population = []

    for _ in range(num_offsprings):
        current = copy.deepcopy(optimal)
        num_moves = random.randint(1, max_moves)
        for _ in range(num_moves):
            possible_moves = generate_all_moves(move_func, current.values)
            if possible_moves:
                next_arr = possible_moves[0]
                current.set_values(next_arr)
            else:
                break
        current.metrics = current.calculate_metrics()
        perturbed_population.append(current)
    return perturbed_population


def calculate_all_similarities_metrics(
    opt: Any, population: Iterable[Any]
) -> List[Tuple[float, Dict[str, float]]]:
    similarities = []
    opt_values = opt.values

    for individual in population:
        sim = calculate_similarity_permutation(opt_values, individual.values)
        similarities.append((sim, individual.metrics))

    return similarities


def calculate_similarity_permutation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    flat1 = arr1.flatten()
    flat2 = arr2.flatten()
    return float(np.sum(flat1 == flat2)) / len(flat1)
