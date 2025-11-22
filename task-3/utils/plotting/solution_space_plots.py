from typing import Any, Dict, Callable
import os
import matplotlib.pyplot as plt
from itertools import combinations
import plotly.graph_objects as go


def plot_similarity_metric_landscape(
    all_similarities: Dict[str, Any],
    optimal: Any,
    analysis_dir: str,
    metric_functions: Dict[str, Callable],
    moves: Dict[str, Callable],
) -> None:
    os.makedirs(analysis_dir, exist_ok=True)

    metric_names = list(metric_functions.keys())
    if not optimal.metrics:
        optimal.metrics = optimal.calculate_metrics()
    optimal_metrics = optimal.metrics

    for move_name in moves.keys():
        fig, axes = plt.subplots(
            nrows=len(metric_names),
            ncols=1,
            figsize=(8, 5 * len(metric_names)),
            squeeze=False,
        )

        for i, metric_name in enumerate(metric_names):
            ax = axes[i][0]
            ax.set_title(f"Move: {move_name} | Metric: {metric_name}")
            ax.set_xlabel("Similarity to Optimal")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

            # Random
            sims_r = [s for s, mv in all_similarities["Random"][metric_name][move_name]]
            vals_r = [
                mv[metric_name]
                for _, mv in all_similarities["Random"][metric_name][move_name]
            ]
            ax.scatter(sims_r, vals_r, s=6, color="blue", alpha=0.6, label="Random")

            # Perturbed
            sims_p = [
                s for s, mv in all_similarities["Perturbed"][metric_name][move_name]
            ]
            vals_p = [
                mv[metric_name]
                for _, mv in all_similarities["Perturbed"][metric_name][move_name]
            ]
            ax.scatter(
                sims_p, vals_p, s=15, color="purple", alpha=0.7, label="Perturbed"
            )

            # Optimal
            ax.scatter(
                1.0,
                optimal_metrics[metric_name],
                s=80,
                color="red",
                edgecolor="black",
                label="Optimal",
            )

            ax.legend(loc="best")

        plt.tight_layout()
        out_path = os.path.join(analysis_dir, f"{move_name}_landscape.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"Saved: {out_path}")


def plot_similarity_metric_3d_landscapes_interactive(
    all_similarities: Dict[str, Any],
    optimal: Any,
    analysis_dir: str,
    metric_functions: Dict[str, Callable],
    moves: Dict[str, Callable],
) -> None:
    os.makedirs(analysis_dir, exist_ok=True)

    metric_names = list(metric_functions.keys())

    if not optimal.metrics:
        optimal.metrics = optimal.calculate_metrics()
    optimal_metrics = optimal.metrics
    metric_pairs = list(combinations(metric_names, 2))

    def extract_xyz(strategy_name, metric_x, metric_y, move_name):
        """Return lists for X, Y, Z."""
        entries = all_similarities[strategy_name][metric_x][move_name]
        sims = []
        xs = []
        ys = []
        for similarity, mv in entries:
            sims.append(similarity)
            xs.append(mv[metric_x])
            ys.append(mv[metric_y])
        return xs, ys, sims

    for move_name in moves.keys():
        for metric_x, metric_y in metric_pairs:

            fig = go.Figure()

            for strategy, color, size in [
                ("Random", "blue", 3),
                ("Perturbed", "purple", 4),
            ]:
                xs, ys, zs = extract_xyz(strategy, metric_x, metric_y, move_name)

                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="markers",
                        marker=dict(size=size, color=color, opacity=0.7),
                        name=strategy,
                        hovertemplate=(
                            f"<b>{strategy}</b><br>"
                            f"{metric_x}: %{{x}}<br>"
                            f"{metric_y}: %{{y}}<br>"
                            f"Similarity: %{{z}}<extra></extra>"
                        ),
                    )
                )

            fig.add_trace(
                go.Scatter3d(
                    x=[optimal_metrics[metric_x]],
                    y=[optimal_metrics[metric_y]],
                    z=[1.0],
                    mode="markers",
                    marker=dict(
                        size=10, color="red", line=dict(width=2, color="black")
                    ),
                    name="Optimal",
                    hovertemplate=(
                        f"<b>Optimal</b><br>"
                        f"{metric_x}: %{{x}}<br>"
                        f"{metric_y}: %{{y}}<br>"
                        f"Similarity: %{{z}}<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title=(
                    f"Interactive 3D Landscape<br>"
                    f"Move = {move_name} | Metrics: {metric_x} vs {metric_y}"
                ),
                scene=dict(
                    xaxis_title=metric_x,
                    yaxis_title=metric_y,
                    zaxis_title="Similarity to Optimal",
                ),
                width=900,
                height=750,
                showlegend=True,
            )

            out_path = os.path.join(
                analysis_dir, f"{move_name}_3D_{metric_x}_vs_{metric_y}.html"
            )

            fig.write_html(out_path, include_plotlyjs="cdn")
            print(f"Saved interactive plot: {out_path}")
