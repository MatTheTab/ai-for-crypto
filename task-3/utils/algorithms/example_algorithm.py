import numpy as np
from utils.SBox import SBox
from utils.Metric import Metric
from utils.initialization import initialize_random
from tqdm import tqdm


def example_algorithm(
    pop_size: int, epochs: int
) -> tuple[SBox, list[SBox], list[list[dict[str:Metric]]]]:
    history = []
    best_ever = None
    final_population = []
    population = initialize_random(pop_size)
    for epoch in tqdm(range(epochs)):
        history_epoch = []
        for individual in population:
            history_epoch.append(individual.metrics)
        history.append(history_epoch)
    best_ever = population[0]
    final_population = population
    return best_ever, final_population, history
