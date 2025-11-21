import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Any


def plot_history(
    best_ever: object,
    final_population: List[Any],
    history: List[List[float]],
    save_dir: str,
    save_filename: str,
) -> None:

    # Create a fresh figure BEFORE plotting
    plt.figure(figsize=(10, 8))

    # Compute bounds once
    maximum = np.max(np.array(history))
    minimum = np.min(np.array(history))

    # Plot each individual's history
    for individual_history in history:
        plt.plot(individual_history, alpha=0.3)

    plt.ylim(minimum * 0.8, maximum * 1.2)
    plt.ylabel("Nonlinearity Value")
    plt.xlabel("Iteration")
    plt.title("Evaluation History")

    save_path = os.path.join(save_dir, save_filename + ".png")
    plt.savefig(save_path)
    plt.close()  # Prevent leaking figures

    print(f"Saved history plot: {save_path}")


def plot_population(
    best_ever: object,
    final_population: List[Any],
    history: List[List[float]],
    save_dir: str,
    save_filename: str,
) -> None:

    metric_names = list(best_ever.metrics.keys())
    if len(metric_names) > 3:
        metric_names = metric_names[:3]

    # --- 2D SCATTER --------------------------------------------
    if len(metric_names) == 2:

        # Fresh figure
        plt.figure(figsize=(8, 6))

        m1, m2, colors = [], [], []
        for individual in final_population:
            m1.append(individual.metrics[metric_names[0]])
            m2.append(individual.metrics[metric_names[1]])

            if (
                individual.metrics[metric_names[0]]
                == best_ever.metrics[metric_names[0]]
                and individual.metrics[metric_names[1]]
                == best_ever.metrics[metric_names[1]]
            ):
                colors.append("red")
            else:
                colors.append("blue")

        plt.scatter(m1, m2, c=colors, alpha=0.5, s=20)
        plt.xlabel(metric_names[0])
        plt.ylabel(metric_names[1])
        plt.title("Final Population Metrics")

        out_path = os.path.join(save_dir, save_filename + ".png")
        plt.savefig(out_path)
        plt.close()

        print(f"Saved population plot: {out_path}")
        return

    # --- 3D PLOTLY PLOT --------------------------------------------
    m1, m2, m3, colors = [], [], [], []
    for individual in final_population:
        m1.append(individual.metrics[metric_names[0]])
        m2.append(individual.metrics[metric_names[1]])
        m3.append(individual.metrics[metric_names[2]])

        if (
            individual.metrics[metric_names[0]] == best_ever.metrics[metric_names[0]]
            and individual.metrics[metric_names[1]]
            == best_ever.metrics[metric_names[1]]
            and individual.metrics[metric_names[2]]
            == best_ever.metrics[metric_names[2]]
        ):
            colors.append("red")
        else:
            colors.append("blue")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=m1,
            y=m2,
            z=m3,
            mode="markers",
            marker=dict(color=colors, opacity=0.7, size=3),
            hovertemplate=(
                f"<b>Individual</b><br>"
                f"{metric_names[0]}: %{{x}}<br>"
                f"{metric_names[1]}: %{{y}}<br>"
                f"{metric_names[2]}: %{{z}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Interactive 3D Population Plot",
        scene=dict(
            xaxis_title=metric_names[0],
            yaxis_title=metric_names[1],
            zaxis_title=metric_names[2],
        ),
        width=900,
        height=750,
    )

    out_path = os.path.join(save_dir, save_filename + ".html")
    fig.write_html(out_path, include_plotlyjs="cdn")

    print(f"Saved interactive plot: {out_path}")
