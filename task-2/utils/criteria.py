import os
import numpy as np
import random
from typing import Callable, Tuple, List, Set, Any, Optional
from deap import base, creator, tools
from utils.utils import (
    xor_bytes,
    get_elementary_bit_vector,
    bytes_to_bitstring,
    hamming_distance_int,
    bytes_to_int,
    change_bit,
    generate_bytes,
    generate_bytes_random_colision,
)
import sys


ITERATION_LIMIT = 2**24


def analyze_results_BIC(results_delta: np.ndarray) -> np.ndarray:
    """
    Calculates the correlation matrix (BIC criterion) for a matrix of bit-wise differences.
    """
    return np.nan_to_num(np.corrcoef(results_delta, rowvar=False), nan=0.0)


def analyze_results_SAC(
    arr_original: np.ndarray, arr_changed: np.ndarray, arr_delta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyzes the Strict Avalanche Criterion (SAC) for bit-wise changes
    and calculates statistical properties of the bit changes.
    """
    message_size = arr_original.shape[-1]
    total_sum = np.sum(arr_original, axis=0)
    results_matrix = np.zeros(shape=(message_size, message_size))
    bit_means_changed = np.mean(arr_changed, axis=0)
    bit_std_changed = np.std(arr_changed, axis=0)
    bit_means_og = np.mean(arr_original, axis=0)
    bit_std_og = np.std(arr_original, axis=0)

    for og_idx in range(message_size):
        for changed_idx in range(message_size):
            result = (
                np.sum(
                    (arr_delta[:, changed_idx] == 1) & (arr_original[:, og_idx] == 1)
                )
                / total_sum[og_idx]
            )
            results_matrix[og_idx][changed_idx] = result
    return results_matrix, bit_means_changed, bit_std_changed, bit_means_og, bit_std_og


def test_SAC_BIC(
    data: List[bytes],
    message_size: int,
    testing_size: int,
    hash_func: Callable[[bytes], bytes],
) -> List[Tuple[str, str, str]]:
    results = []
    for datum in data:
        hashed = hash_func(datum)
        for i in range(testing_size):
            ei = get_elementary_bit_vector(i, message_size * 8)
            changed = xor_bytes(datum, ei)
            hashed_changed = hash_func(changed)
            delta = xor_bytes(hashed, hashed_changed)

            hashed_bits = bytes_to_bitstring(hashed)[-testing_size:]
            hashed_changed_bits = bytes_to_bitstring(hashed_changed)[-testing_size:]
            delta_bits = bytes_to_bitstring(delta)[-testing_size:]
            results.append((hashed_bits, hashed_changed_bits, delta_bits))

    return results


def greedy_colision_detection(
    byte_range: int,
    test_length: int,
    hashing_algorithm: Callable[[bytes], bytes],
    num_bytes: int,
    retry: bool = True,
) -> Tuple[int, bool, List[int]]:
    """
    Attempts to find a hash collision with a target hash using a greedy local search algorithm,
    minimizing Hamming distance to the target hash.
    """
    hamming_distances = []
    num_bits = num_bytes * 8
    total_searches = 0
    target = os.urandom(num_bytes)
    target_hash = hashing_algorithm(target)[:test_length]

    curr_solution = os.urandom(num_bytes)
    while curr_solution == target:
        curr_solution = os.urandom(num_bytes)

    curr_hash = hashing_algorithm(curr_solution)[:test_length]
    if curr_hash == target_hash:
        return 0, True, [0]

    best_so_far = curr_solution
    best_value_so_far = hamming_distance_int(
        bytes_to_int(curr_hash), bytes_to_int(target_hash)
    )
    hamming_distances.append(best_value_so_far)
    total_searches += 1

    for search_idx in range(byte_range):
        solution_changed = False
        for bit_change_idx in range(num_bits):
            if total_searches > ITERATION_LIMIT:
                break

            total_searches += 1
            new_solution = change_bit(curr_solution, bit_change_idx)
            new_hash = hashing_algorithm(new_solution)[:test_length]
            new_val = hamming_distance_int(
                bytes_to_int(new_hash), bytes_to_int(target_hash)
            )

            if new_val < best_value_so_far:
                best_value_so_far = new_val
                best_so_far = new_solution
                solution_changed = True
                break

        if solution_changed:
            curr_solution = best_so_far
            hamming_distances.append(best_value_so_far)
            if best_so_far == 0:
                return total_searches, True, hamming_distances

        elif retry:
            curr_solution = os.urandom(num_bytes)
            curr_hash = hashing_algorithm(curr_solution)[:test_length]
            total_searches += 1
            best_value_so_far = hamming_distance_int(
                bytes_to_int(curr_hash), bytes_to_int(target_hash)
            )
            hamming_distances.append(best_value_so_far)
            if curr_hash == target_hash:
                return total_searches, True, hamming_distances
        else:
            break
    return total_searches, False, hamming_distances


def steepest_colision_detection(
    byte_range: int,
    test_length: int,
    hashing_algorithm: Callable[[bytes], bytes],
    num_bytes: int,
    retry: bool = True,
) -> Tuple[int, bool, List[int]]:
    """
    Attempts to find a hash collision with a target hash using the steepest descent local search algorithm,
    minimizing Hamming distance to the target hash.
    """
    hamming_distances = []
    num_bits = num_bytes * 8
    total_searches = 0
    target = os.urandom(num_bytes)
    target_hash = hashing_algorithm(target)[:test_length]

    curr_solution = os.urandom(num_bytes)
    while curr_solution == target:
        curr_solution = os.urandom(num_bytes)

    curr_hash = hashing_algorithm(curr_solution)[:test_length]
    if curr_hash == target_hash:
        return 0, True, [0]

    best_so_far = curr_solution
    best_value_so_far = hamming_distance_int(
        bytes_to_int(curr_hash), bytes_to_int(target_hash)
    )
    hamming_distances.append(best_value_so_far)
    total_searches += 1

    for search_idx in range(byte_range):
        solution_changed = False
        for bit_change_idx in range(num_bits):
            if total_searches > ITERATION_LIMIT:
                break

            total_searches += 1
            new_solution = change_bit(curr_solution, bit_change_idx)
            new_hash = hashing_algorithm(new_solution)[:test_length]
            new_val = hamming_distance_int(
                bytes_to_int(new_hash), bytes_to_int(target_hash)
            )

            if new_val < best_value_so_far:
                best_value_so_far = new_val
                best_so_far = new_solution
                solution_changed = True

        if solution_changed:
            curr_solution = best_so_far
            hamming_distances.append(best_value_so_far)
            if best_so_far == 0:
                return total_searches, True, hamming_distances

        elif retry:
            curr_solution = os.urandom(num_bytes)
            curr_hash = hashing_algorithm(curr_solution)[:test_length]
            total_searches += 1
            best_value_so_far = hamming_distance_int(
                bytes_to_int(curr_hash), bytes_to_int(target_hash)
            )
            hamming_distances.append(best_value_so_far)
            if curr_hash == target_hash:
                return total_searches, True, hamming_distances
        else:
            break
    return total_searches, False, hamming_distances


def bruteforce_colision_detection(
    test_length: int,
    byte_range: int,
    num_bytes: int,
    hashing_algorithm: Callable[[bytes], bytes],
    mode: str = "any",
) -> Tuple[int, bool]:
    """
    Attempts to find a hash collision (first-preimage or second-preimage) using brute-force search.
    """
    if mode not in ("any", "one"):
        raise ValueError("Incorrect mode")

    all_hashes = set([])
    if mode == "one":
        target = os.urandom(num_bytes)
        target_hash = hashing_algorithm(target)
        target = target_hash[:test_length]
        all_hashes.add(target)

    for search_idx, byte_data in enumerate(
        generate_bytes(byte_range=byte_range, num_bytes=num_bytes)
    ):
        if search_idx > ITERATION_LIMIT:
            break
        hashed = hashing_algorithm(byte_data)
        cut_hash = hashed[:test_length]

        if cut_hash in all_hashes:
            return search_idx, True

        if mode == "any":
            all_hashes.add(cut_hash)

    return search_idx, False


def random_colision_detection(
    test_length: int,
    num_bytes: int,
    hashing_algorithm: Callable[[bytes], bytes],
    mode: str = "any",
) -> Tuple[int, bool]:
    """
    Attempts to find a hash collision (first-preimage or second-preimage) using random message generation.
    """
    if mode not in ("any", "one"):
        raise ValueError("Incorrect mode")

    all_hashes = set([])
    if mode == "one":
        target = os.urandom(num_bytes)
        target_hash = hashing_algorithm(target)[:test_length]
        all_hashes.add(target_hash)

    for search_idx, byte_data in enumerate(
        generate_bytes_random_colision(num_bytes=num_bytes)
    ):
        if search_idx > ITERATION_LIMIT:
            break
        hashed = hashing_algorithm(byte_data)
        cut_hash = hashed[:test_length]

        if cut_hash in all_hashes:
            return search_idx, True

        if mode == "any":
            all_hashes.add(cut_hash)

    return search_idx, False


def genetic_colision_search(
    test_length: int,
    num_bytes: int,
    hashing_algorithm: Callable[[bytes], bytes],
    pop_size: int,
    num_generations: int,
    prob_mutation: float,
    prob_crossover: float,
    torunament_size: int,
) -> Tuple[int, bool, List[int], List[float], int]:
    """
    Searches for a second-preimage hash collision using a Genetic Algorithm (GA),
    minimizing Hamming distance to a target hash.
    """
    target = os.urandom(num_bytes)
    target_hash = hashing_algorithm(target)[:test_length]
    hamming_distances = []
    hamming_distances_mean = []

    def hashing_fitness(individual):
        bit_string = "".join(str(b) for b in individual)
        value = int(bit_string, 2)
        byte_repr = value.to_bytes(num_bytes, byteorder=sys.byteorder)
        individual_hash = hashing_algorithm(byte_repr)[:test_length]
        value = hamming_distance_int(
            bytes_to_int(individual_hash), bytes_to_int(target_hash)
        )
        return (-1 * value,)

    num_bits = num_bytes * 8
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_bits
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", hashing_fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)
    toolbox.register("select", tools.selTournament, tournsize=torunament_size)
    population = toolbox.population(n=pop_size)

    for gen in range(num_generations):
        if gen * pop_size > ITERATION_LIMIT:
            break
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < prob_crossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < prob_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        best_fitness = max(fits)
        best_fitness *= -1
        hamming_distances.append(best_fitness)
        hamming_distances_mean.append(sum(fits) / len(fits))
        if max(fits) == 0:
            break

    best_ind = tools.selBest(population, 1)[0]
    best_fitness = best_ind.fitness.values[0]
    if best_fitness == 0:
        return gen * pop_size, True, hamming_distances, hamming_distances_mean, gen
    return gen * pop_size, False, hamming_distances, hamming_distances_mean, gen


def test_colision_detection(
    num_tests: int,
    colision_detection_algorithm: str,
    test_length: int,
    byte_range: int,
    num_bytes: int,
    hashing_algorithm: Callable[[bytes], bytes],
    mode: Optional[str] = None,
    pop_size: Optional[int] = None,
    num_generations: Optional[int] = None,
    prob_mutation: Optional[float] = None,
    prob_crossover: Optional[float] = None,
    torunament_size: Optional[int] = None,
    pretty_print: bool = True,
    alg_name: str = "",
) -> Tuple[
    List[int],
    List[bool],
    List[Optional[List[int]]],
    List[Optional[List[float]]],
    List[Optional[int]],
]:
    """
    Runs multiple tests for different collision detection algorithms and aggregates the results.
    """
    results = []
    statuses = []
    distances = []
    distances_mean = []
    generations = []

    for i in range(num_tests):
        if colision_detection_algorithm == "bruteforce":
            result, status = bruteforce_colision_detection(
                test_length=test_length,
                byte_range=byte_range,
                num_bytes=num_bytes,
                hashing_algorithm=hashing_algorithm,
                mode=mode,
            )
            hamming_distances = None
            mean_distances = None
            generation = None

        elif colision_detection_algorithm == "random":
            result, status = random_colision_detection(
                test_length=test_length,
                num_bytes=num_bytes,
                hashing_algorithm=hashing_algorithm,
                mode=mode,
            )
            hamming_distances = None
            mean_distances = None
            generation = None

        elif colision_detection_algorithm == "steepest":
            result, status, hamming_distances = steepest_colision_detection(
                test_length=test_length,
                byte_range=byte_range,
                num_bytes=num_bytes,
                hashing_algorithm=hashing_algorithm,
                retry=True,
            )
            mean_distances = None
            generation = None

        elif colision_detection_algorithm == "greedy":
            result, status, hamming_distances = greedy_colision_detection(
                test_length=test_length,
                byte_range=byte_range,
                num_bytes=num_bytes,
                hashing_algorithm=hashing_algorithm,
                retry=True,
            )
            mean_distances = None
            generation = None

        elif colision_detection_algorithm == "genetic":
            result, status, hamming_distances, mean_distances, generation = (
                genetic_colision_search(
                    test_length=test_length,
                    num_bytes=num_bytes,
                    hashing_algorithm=hashing_algorithm,
                    pop_size=pop_size,
                    num_generations=num_generations,
                    prob_mutation=prob_mutation,
                    prob_crossover=prob_crossover,
                    torunament_size=torunament_size,
                )
            )

        results.append(result)
        statuses.append(status)
        distances.append(hamming_distances)
        distances_mean.append(mean_distances)
        generations.append(generation)

    if pretty_print:
        print(
            f"{colision_detection_algorithm} for algorithm: {alg_name} detection of colision mean iterations: {np.mean(results)} standard deviation: {np.std(results)}"
        )
    return results, statuses, distances, distances_mean, generations
