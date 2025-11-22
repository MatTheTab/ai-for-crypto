import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_histories(all_histories, save_dir="plots"):
    """
    Plots the histories of nonlinearity values over iterations.
    Each algorithm is assigned a color. All runs are plotted as lines with alpha.
    """
    ensure_dir(save_dir)

    plt.figure(figsize=(12, 8))

    # Get a colormap to generate unique colors for each algorithm
    algorithms = list(all_histories.keys())
    colors = plt.cm.get_cmap("tab10", len(algorithms))

    for idx, (algo_name, runs) in enumerate(all_histories.items()):
        color = colors(idx)
        for run_idx, run_data in enumerate(runs):
            # Only add label to the first run of the algorithm to avoid legend clutter
            label = algo_name if run_idx == 0 else None

            plt.plot(run_data, color=color, alpha=0.3, linewidth=1.5, label=label)

    plt.title("Non-Linearity History per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Non-Linearity")
    plt.legend(title="Algorithm")
    plt.grid(True, linestyle="--", alpha=0.8)

    save_path = os.path.join(save_dir, "histories_nonlinearities.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved histories plot to {save_path}")


def plot_times(all_times, save_dir="plots"):
    """
    Plots a violin plot comparing execution times of different algorithms.
    """
    ensure_dir(save_dir)

    # Convert dictionary structure to DataFrame for Seaborn
    data = []
    for algo_name, times in all_times.items():
        for t in times:
            data.append({"Algorithm": algo_name, "Time (s)": t})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="Algorithm", y="Time (s)", palette="muted")

    plt.title("Execution Time Distribution by Algorithm")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")

    save_path = os.path.join(save_dir, "execution_times_violin.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved times plot to {save_path}")


def plot_metrics_scatter(all_metrics, metrics_to_plot, save_dir="plots"):
    """
    Creates scatterplots based on the number of metrics provided.
    - 2 Metrics: 2D Static PNG.
    - 3+ Metrics: Interactive 3D HTML (Plotly).
      - 3 Metrics: X, Y, Z.
      - 4 Metrics: Size encodes 4th metric.
      - 5 Metrics: Symbol encodes 5th metric (to preserve Color for Algorithm).
    """
    ensure_dir(save_dir)

    # Flatten data into a DataFrame
    rows = []
    for algo_name, runs in all_metrics.items():
        for run_metrics in runs:
            row = {"Algorithm": algo_name}
            # Add all requested metrics to the row
            for m in metrics_to_plot:
                # Safe get in case a metric is missing from a specific run
                row[m] = run_metrics.get(m, None)
            rows.append(row)

    df = pd.DataFrame(rows)

    num_metrics = len(metrics_to_plot)

    if num_metrics == 2:
        # 2D Static Plot using Seaborn
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df,
            x=metrics_to_plot[0],
            y=metrics_to_plot[1],
            hue="Algorithm",
            style="Algorithm",
            s=100,
            alpha=0.8,
        )
        plt.title(f"{metrics_to_plot[0]} vs {metrics_to_plot[1]}")

        save_path = os.path.join(save_dir, "metrics_scatter_2d.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved 2D scatter plot to {save_path}")

    elif num_metrics >= 3:
        # 3D Interactive Plot using Plotly
        x_col = metrics_to_plot[0]
        y_col = metrics_to_plot[1]
        z_col = metrics_to_plot[2]

        size_col = metrics_to_plot[3] if num_metrics >= 4 else None
        # Note: We use symbol for the 5th metric because 'color' is strictly reserved
        # for Algorithm identity per requirements.
        symbol_col = metrics_to_plot[4] if num_metrics >= 5 else None

        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color="Algorithm",
            size=size_col,
            symbol=symbol_col,
            hover_data=metrics_to_plot,
            title="Multi-Metric Algorithm Comparison",
        )

        # Improve visibility
        fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))

        # Save as interactive HTML
        save_path = os.path.join(save_dir, "metrics_scatter_3d.html")
        fig.write_html(save_path)
        print(f"Saved interactive 3D scatter plot to {save_path}")
    else:
        print("Please provide at least 2 metrics to plot.")
