from typing import Any, Dict, List, Tuple, Optional, Iterable, Callable
from utils.initialization import initialize_random
from utils.algorithms.alg_utils import find_best_ever
import random
import numpy as np


def generate_all_moves(
    move: Callable[[np.ndarray], Iterable[np.ndarray]],
    arr: np.ndarray,
) -> List[np.ndarray]:
    moves = list(move(arr))
    random.shuffle(moves)
    return moves


def greedy(
    population: List[Any],
    metric_name: str,
    optimization_direction: str,
    move: Callable[[np.ndarray], Iterable[np.ndarray]],
    move_limit: Optional[int] = None,
) -> Tuple[Optional[Any], List[Any], List[List[float]]]:
    new_population: List[Any] = []
    history: List[List[float]] = []

    maximize = optimization_direction == "maximize"

    for individual in population:
        current = individual.copy()
        indiv_history: List[float] = []

        while True:
            metric_val = current.metrics[metric_name]
            indiv_history.append(metric_val)

            all_moves = generate_all_moves(move, current.values)
            if move_limit:
                all_moves = all_moves[:move_limit]

            improved = False
            for new_arr in all_moves:
                new_val = current.evaluate_with_values(new_arr, metric_name)

                if (maximize and new_val > metric_val) or (
                    not maximize and new_val < metric_val
                ):
                    current = current.make_new(new_arr)
                    if not current.metrics or metric_name not in current.metrics:
                        current.metrics = current.calculate_metrics()

                    improved = True
                    break

            if not improved:
                new_population.append(current)
                history.append(indiv_history)
                break

    best = find_best_ever(new_population, metric_name, optimization_direction)
    return best, new_population, history


def steepest(
    population: List[Any],
    metric_name: str,
    optimization_direction: str,
    move: Callable[[np.ndarray], Iterable[np.ndarray]],
    move_limit: Optional[int] = None,
) -> Tuple[Optional[Any], List[Any], List[List[float]]]:
    new_population: List[Any] = []
    history: List[List[float]] = []

    maximize = optimization_direction == "maximize"

    for individual in population:
        current = individual.copy()
        indiv_history: List[float] = []

        while True:
            metric_val = current.metrics[metric_name]
            indiv_history.append(metric_val)

            best_val = metric_val
            best_arr: Optional[np.ndarray] = None

            all_moves = generate_all_moves(move, current.values)
            if move_limit:
                all_moves = all_moves[:move_limit]

            for new_arr in all_moves:
                new_val = current.evaluate_with_values(new_arr, metric_name)

                if (maximize and new_val > best_val) or (
                    not maximize and new_val < best_val
                ):
                    best_val = new_val
                    best_arr = new_arr

            if best_arr is None:
                new_population.append(current)
                history.append(indiv_history)
                break

            current = current.make_new(best_arr)
            if not current.metrics or metric_name not in current.metrics:
                current.metrics = current.calculate_metrics()

    best = find_best_ever(new_population, metric_name, optimization_direction)
    return best, new_population, history


def node_swap_move(arr: np.ndarray) -> Iterable[np.ndarray]:
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            new_arr = arr.copy()
            new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
            yield new_arr


def edge_swap_move(arr: np.ndarray) -> Iterable[np.ndarray]:
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            new_arr = arr.copy()
            new_arr[i : j + 1] = new_arr[i : j + 1][::-1]
            yield new_arr
