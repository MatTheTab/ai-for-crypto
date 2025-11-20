import matplotlib.pyplot as plt
from utils.SBox import SBox
from utils.Metric import Metric
import os


def save_plot(save_dir: str, save_filename: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    full_path = os.path.join(save_dir, save_filename)
    plt.savefig(full_path, format="png", dpi=300)


def example_plot(
    best_ever: SBox,
    population: list[SBox],
    metrics: list[list[dict[str:Metric]]],
    save_dir: str,
    save_filename: str,
) -> None:
    plt.plot([1, 2, 3, 4])
    save_plot(save_dir=save_dir, save_filename=save_filename)
