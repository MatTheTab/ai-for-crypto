import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from scipy.stats import gaussian_kde


def plot_SAC_results(
    all_results: Dict[
        str,
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
    figsize: Tuple[int, int] = (18, 10),
    title: str = "SAC Analysis Results",
    x_label: str = "Changed Bit Index",
    y_label: str = "Original Bit Index",
    annot_values: bool = False,
    normalize: bool = True,
) -> None:
    """Plots results from Strict Avalanche Criterion (SAC) analysis."""

    n_algorithms = len(all_results)
    n_rows = 5

    fig, axes = plt.subplots(
        n_rows,
        n_algorithms,
        figsize=figsize,
        gridspec_kw={"height_ratios": [6, 0.5, 0.5, 0.5, 0.5]},
    )
    if n_algorithms == 1:
        axes = np.expand_dims(axes, axis=1)

    cmap_matrix = "mako"
    cmap_mean_og = "crest"
    cmap_std_og = "crest"
    cmap_mean_ch = "rocket"
    cmap_std_ch = "rocket"

    if normalize:
        matrices, mean_og, std_og, mean_ch, std_ch = [], [], [], [], []
        for values in all_results.values():
            m, mch, sch, mog, sog, corr = values
            matrices.append(m)
            mean_ch.append(mch)
            std_ch.append(sch)
            mean_og.append(mog)
            std_og.append(sog)

        def global_min_max(arrs):
            all_vals = np.concatenate([a.flatten() for a in arrs])
            return np.min(all_vals), np.max(all_vals)

        vmin_matrix, vmax_matrix = global_min_max(matrices)
        vmin_mean_og, vmax_mean_og = global_min_max(mean_og)
        vmin_std_og, vmax_std_og = global_min_max(std_og)
        vmin_mean_ch, vmax_mean_ch = global_min_max(mean_ch)
        vmin_std_ch, vmax_std_ch = global_min_max(std_ch)
    else:
        vmin_matrix = vmax_matrix = None
        vmin_mean_og = vmax_mean_og = None
        vmin_std_og = vmax_std_og = None
        vmin_mean_ch = vmax_mean_ch = None
        vmin_std_ch = vmax_std_ch = None

    for col_idx, (alg_name, values) in enumerate(all_results.items()):
        (
            results_matrix,
            bit_means_changed,
            bit_std_changed,
            bit_means_og,
            bit_std_og,
            corr,
        ) = values

        sns.heatmap(
            results_matrix,
            ax=axes[0, col_idx],
            cmap=cmap_matrix,
            cbar=True,
            vmin=vmin_matrix,
            vmax=vmax_matrix,
            annot=annot_values,
            fmt=".2f" if annot_values else "",
            annot_kws={"size": 6},
        )
        axes[0, col_idx].set_title(alg_name, fontsize=14, weight="bold", pad=10)
        axes[0, col_idx].set_xlabel(x_label, fontsize=10)
        axes[0, col_idx].set_ylabel(y_label if col_idx == 0 else "", fontsize=10)

        sns.heatmap(
            bit_means_og[np.newaxis, :],
            ax=axes[1, col_idx],
            cmap=cmap_mean_og,
            cbar=False,
            xticklabels=False,
            yticklabels=["Mean (OG)"],
            vmin=vmin_mean_og,
            vmax=vmax_mean_og,
            annot=annot_values,
            fmt=".2f" if annot_values else "",
            annot_kws={"size": 6},
        )

        sns.heatmap(
            bit_std_og[np.newaxis, :],
            ax=axes[2, col_idx],
            cmap=cmap_std_og,
            cbar=False,
            xticklabels=False,
            yticklabels=["Std (OG)"],
            vmin=vmin_std_og,
            vmax=vmax_std_og,
            annot=annot_values,
            fmt=".2f" if annot_values else "",
            annot_kws={"size": 6},
        )

        sns.heatmap(
            bit_means_changed[np.newaxis, :],
            ax=axes[3, col_idx],
            cmap=cmap_mean_ch,
            cbar=False,
            xticklabels=False,
            yticklabels=["Mean (Changed)"],
            vmin=vmin_mean_ch,
            vmax=vmax_mean_ch,
            annot=annot_values,
            fmt=".2f" if annot_values else "",
            annot_kws={"size": 6},
        )

        sns.heatmap(
            bit_std_changed[np.newaxis, :],
            ax=axes[4, col_idx],
            cmap=cmap_std_ch,
            cbar=False,
            xticklabels=False,
            yticklabels=["Std (Changed)"],
            vmin=vmin_std_ch,
            vmax=vmax_std_ch,
            annot=annot_values,
            fmt=".2f" if annot_values else "",
            annot_kws={"size": 6},
        )

        for row in range(1, n_rows):
            axes[row, col_idx].tick_params(left=False, bottom=False)
            if col_idx != 0:
                axes[row, col_idx].set_ylabel("")

    fig.suptitle(title, fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_BIC_results(
    all_results: Dict[str, Tuple[Any, Any, Any, Any, Any, np.ndarray]],
    figsize: Tuple[int, int] = (18, 10),
    title: str = "Bit Independence Criterion (BIC) Analysis",
    x_label: str = "Output Bit Index",
    y_label: str = "Output Bit Index",
    annot_values: bool = False,
    normalize: bool = True,
) -> None:
    """Plots results from Bit Independence Criterion (BIC) analysis."""

    n_algorithms = len(all_results)
    fig, axes = plt.subplots(1, n_algorithms, figsize=figsize)

    if n_algorithms == 1:
        axes = [axes]

    corr_matrices = [vals[-1] for vals in all_results.values()]
    if normalize:
        all_vals = np.concatenate([cm.flatten() for cm in corr_matrices])
        vmin, vmax = np.min(all_vals), np.max(all_vals)
    else:
        vmin = vmax = None

    cmap_corr = sns.diverging_palette(230, 20, as_cmap=True)
    for ax, (alg_name, vals) in zip(axes, all_results.items()):
        corr_matrix = vals[-1]

        sns.heatmap(
            corr_matrix,
            ax=ax,
            cmap=cmap_corr,
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=True,
            annot=annot_values,
            fmt=".2f" if annot_values else "",
            annot_kws={"size": 6},
        )
        ax.set_title(alg_name, fontsize=14, weight="bold", pad=10)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)

    fig.suptitle(title, fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_collision_histograms(
    results_dict: Dict[str, Dict[str, Dict[str, Any]]],
    title: str = "Collision Detection Performance",
    figsize: Tuple[int, int] = (10, 8),
    log_x: bool = False,
    log_y: bool = False,
    smooth: bool = False,
) -> None:
    """Plots histograms (or smooth KDEs) of the number of checks before collision for different algorithms."""

    color_map = {
        "SHA256": "green",
        "MD5": "blue",
        "Random": "red",
        "Custom": "orange",
    }

    algorithms = list(results_dict.keys())
    n_rows = len(algorithms)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)

    if n_rows == 1:
        axes = [axes]

    for ax, algorithm in zip(axes, algorithms):
        subdict = results_dict[algorithm]
        all_vals = []

        # Gather all data for global per-algorithm range
        for hash_name, data in subdict.items():
            vals = np.array(data.get("Num Checked Solutions", []), dtype=float)
            if len(vals) > 0:
                all_vals.extend(vals)

        if not all_vals:
            continue

        global_min, global_max = np.min(all_vals), np.max(all_vals)
        if global_min == global_max:
            # artificially expand range to make visible
            global_min -= 0.5
            global_max += 0.5

        x_range = np.linspace(global_min, global_max, 300)

        # Plot each hash variant
        for hash_name, data in subdict.items():
            vals = np.array(data.get("Num Checked Solutions", []), dtype=float)
            if len(vals) == 0:
                continue

            # avoid zero-range collapse
            if np.min(vals) == np.max(vals):
                vals = vals + np.linspace(-0.25, 0.25, len(vals))

            if smooth:
                # Smooth distribution using Gaussian KDE
                kde = gaussian_kde(vals)
                y_vals = kde(x_range)
                y_vals /= y_vals.sum()  # normalize to sum = 1 (relative frequency)
                ax.plot(
                    x_range,
                    y_vals,
                    color=color_map.get(hash_name, "gray"),
                    label=hash_name,
                    lw=2,
                    alpha=0.8,
                )
            else:
                # Standard histogram (relative frequencies)
                counts, bins = np.histogram(
                    vals, bins=20, range=(global_min, global_max)
                )
                rel_freq = counts / counts.sum()

                ax.bar(
                    (bins[:-1] + bins[1:]) / 2,  # bin centers
                    rel_freq,
                    width=(bins[1] - bins[0]) * 0.9,
                    alpha=0.5,
                    color=color_map.get(hash_name, "gray"),
                    label=hash_name,
                    edgecolor="black",
                )

        # Axis styling
        y_label = "Relative\nfrequency" + (" (log)" if log_y else "")
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(f"{algorithm.capitalize()} search", fontsize=12, pad=8)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Visibility and log scaling
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-4)
        else:
            ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 0.05))

    axes[-1].set_xlabel("Number of checks before collision", fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_collision_success_rates(
    results_dict: Dict[str, Dict[str, Dict[str, Any]]],
    title: str = "Collision Success Rates",
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """Plots the collision success rates for different algorithms."""

    color_map = {
        "SHA256": "green",
        "MD5": "blue",
        "Random": "red",
        "Custom": "orange",
    }

    algorithms = list(results_dict.keys())
    n_rows = len(algorithms)

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=False)

    if n_rows == 1:
        axes = [axes]

    for ax, algorithm in zip(axes, algorithms):
        hash_names = []
        success_rates = []
        colors = []

        for hash_name, data in results_dict[algorithm].items():
            statuses = data.get("Statuses", [])
            if len(statuses) == 0:
                continue
            rate = sum(statuses) / len(statuses) * 100
            hash_names.append(hash_name)
            success_rates.append(rate)
            colors.append(color_map.get(hash_name, "gray"))

        ax.bar(hash_names, success_rates, color=colors, alpha=0.7, edgecolor="black")
        ax.set_ylim(0, 100)
        ax.set_title(f"{algorithm.capitalize()} algorithm", fontsize=12, pad=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_genetic_convergence(
    results_dict: Dict[str, Dict[str, Dict[str, Any]]],
    title: str = "Genetic Algorithm Convergence",
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """Plots the convergence of the genetic algorithm, showing best and mean distances."""

    color_map = {
        "SHA256": "green",
        "MD5": "blue",
        "Random": "red",
        "Custom": "orange",
    }
    genetic_data = results_dict.get("genetic", {})

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    best_ax, mean_ax = axes

    for hash_name, data in genetic_data.items():
        all_best = data.get("Closest Distances", [])
        all_mean = data.get("Mean Distances", [])

        for run_best, run_mean in zip(all_best, all_mean):
            if run_best is None or run_mean is None:
                continue
            generations = np.arange(len(run_best))
            best_ax.plot(
                generations, run_best, color=color_map.get(hash_name, "gray"), alpha=0.4
            )
            mean_ax.plot(
                generations, run_mean, color=color_map.get(hash_name, "gray"), alpha=0.4
            )

        if all_best and all_mean:
            max_len_best = max(len(r) for r in all_best if r is not None)
            max_len_mean = max(len(r) for r in all_mean if r is not None)
            best_padded = [
                np.pad(r, (0, max_len_best - len(r)), constant_values=np.nan)
                for r in all_best
                if r is not None
            ]
            mean_padded = [
                np.pad(r, (0, max_len_mean - len(r)), constant_values=np.nan)
                for r in all_mean
                if r is not None
            ]
            best_avg = np.nanmean(best_padded, axis=0)
            mean_avg = np.nanmean(mean_padded, axis=0)
            generations_best = np.arange(len(best_avg))
            generations_mean = np.arange(len(mean_avg))
            best_ax.plot(
                generations_best,
                best_avg,
                color=color_map.get(hash_name, "gray"),
                linewidth=2,
                label=hash_name,
            )
            mean_ax.plot(
                generations_mean,
                mean_avg,
                color=color_map.get(hash_name, "gray"),
                linewidth=2,
                label=hash_name,
            )

    best_ax.set_ylabel("Best distance", fontsize=10)
    mean_ax.set_ylabel("Mean distance", fontsize=10)
    mean_ax.set_xlabel("Generation", fontsize=11)
    best_ax.set_title("Convergence of best individual", fontsize=12, pad=6)
    mean_ax.set_title("Convergence of population mean", fontsize=12, pad=6)
    best_ax.legend(frameon=False)
    best_ax.grid(True, linestyle="--", alpha=0.3)
    mean_ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
