from utils.initialization import initialize_random
from utils.algorithms.alg_utils import find_best_ever_many_criteria
from typing import List


def random_search(
    population_size: int,
    metric_functions: dict,
    metric_names: List[str],
    optimization_directions: List[str],
):
    history = []
    population = initialize_random(
        num_individuals=population_size,
        eager_metric_calculations=True,
        metric_functions=metric_functions,
    )
    best = find_best_ever_many_criteria(
        population, metric_names, optimization_directions
    )
    for individual in population:
        history.append([individual.metrics[metric_names[0]]])
    return best, population, history
