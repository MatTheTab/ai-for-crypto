import numpy as np
import os
from utils.SBox import SBox


def read_sbox_from_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    sbox_line = None
    break_next = False
    for line in lines:
        if break_next:
            sbox_line = line
            break
        if line.strip().startswith("S-box values"):
            break_next = True

    if sbox_line is None:
        raise ValueError("No 'S-box values:' line found in file.")
    values_str = sbox_line.strip()
    values = np.array([int(v.strip()) for v in values_str.split(",")], dtype=np.uint8)
    return values


def read_time_from_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    time_line = None
    for line in lines:
        if line.strip().startswith("Total calculation time:"):
            time_line = line
            break

    if time_line is None:
        raise ValueError("No 'S-box values:' line found in file.")
    values_str = time_line.strip().split(":")[1].split(" ")[1]
    return float(values_str)


def read_history_from_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    values = []
    for line in lines:
        val = float(line.strip())
        values.append(val)
    return values


def find_files_starting_with(folder, prefix):
    """
    Returns a list of full file paths for all files in `folder`
    whose names start with `prefix`.
    """
    matches = []
    for name in os.listdir(folder):
        full_path = os.path.join(folder, name)
        if os.path.isfile(full_path) and name.startswith(prefix):
            matches.append(full_path)
    return matches


def list_direct_subfolders(folder):
    """
    Returns a list of names of direct subfolders in the given folder.
    """
    return [
        name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))
    ]


def parse_results(results_files, metric_functions):
    sboxes = []
    for result in results_files:
        values = read_sbox_from_file(result)
        obj = SBox(
            values=values,
            eager_metric_calculations=True,
            metric_functions=metric_functions,
        )
        sboxes.append(obj.metrics)
    return sboxes


def parse_times(time_files):
    times = []
    for result in time_files:
        value = read_time_from_file(result)
        times.append(value)
    return times


def parse_histories(history_files):
    histories = []
    for result in history_files:
        values = read_history_from_file(result)
        histories.append(values)
    return histories


def parse_all_files(results_dir, metric_functions):
    subfolders = list_direct_subfolders(results_dir)
    all_metrics = {}
    all_times = {}
    all_histories = {}
    for subfolder in subfolders:
        complete_path = os.path.join(results_dir, subfolder)
        results_files = find_files_starting_with(complete_path, "result_")
        all_metrics[subfolder] = parse_results(results_files, metric_functions)

        time_files = find_files_starting_with(complete_path, "time_")
        all_times[subfolder] = parse_times(time_files)

        history_files = find_files_starting_with(complete_path, "best_")
        all_histories[subfolder] = parse_histories(history_files)

    return all_metrics, all_times, all_histories
