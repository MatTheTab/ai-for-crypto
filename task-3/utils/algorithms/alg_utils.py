from typing import Any, List, Optional


def find_best_ever(
    population: List[Any],
    metric_name: str,
    optimization_direction: str,
) -> Optional[Any]:
    if not population:
        return None

    def metric(ind: Any) -> float:
        return ind.metrics[metric_name]

    if optimization_direction == "maximize":
        return max(population, key=metric)
    elif optimization_direction == "minimize":
        return min(population, key=metric)

    raise ValueError("optimization_direction must be 'maximize' or 'minimize'")


def find_best_ever_many_criteria(
    population: List[Any],
    metric_names: List[str],
    optimization_directions: List[str],
) -> Optional[Any]:
    if not population:
        return None
    if len(metric_names) != len(optimization_directions):
        raise ValueError(
            "metric_names and optimization_directions must have the same length"
        )
    candidates = population
    for metric_name, direction in zip(metric_names, optimization_directions):
        values = [ind.metrics[metric_name] for ind in candidates]
        if direction == "maximize":
            best_value = max(values)
        elif direction == "minimize":
            best_value = min(values)
        else:
            raise ValueError("optimization_directions must be 'maximize' or 'minimize'")
        candidates = [
            ind for ind in candidates if ind.metrics[metric_name] == best_value
        ]
        if len(candidates) == 1:
            return candidates[0]
    return candidates[0] if candidates else None
