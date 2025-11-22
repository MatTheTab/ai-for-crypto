import pandas as pd
import os


def calculate_results_statistics(all_metrics, save_csv_path=None):
    """
    Calculates Min, Max, Mean, Median, and Std Dev for every metric
    across every algorithm.

    Args:
        all_metrics (dict): Dictionary of algorithms and their run metrics.
        save_csv_path (str, optional): If provided, saves the stats to a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics.
    """
    stats_rows = []

    for algo_name, runs in all_metrics.items():
        # Convert list of dicts (runs) into a DataFrame for easy math
        df_runs = pd.DataFrame(runs)

        # Iterate over every column (metric) in this algorithm's data
        for metric_name in df_runs.columns:
            values = df_runs[metric_name]

            stats_rows.append(
                {
                    "Algorithm": algo_name,
                    "Metric": metric_name,
                    "Min": values.min(),
                    "Max": values.max(),
                    "Mean": values.mean(),
                    "Median": values.median(),
                    "Std": values.std(),
                }
            )

    # Create the final summary DataFrame
    summary_df = pd.DataFrame(stats_rows)

    # Sort for better readability (group by Algorithm, then Metric)
    summary_df = summary_df.sort_values(by=["Algorithm", "Metric"])

    # Save to CSV if path is provided
    if save_csv_path:
        directory = os.path.dirname(save_csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        summary_df.to_csv(save_csv_path, index=False)
        print(f"Statistics saved to {save_csv_path}")

    return summary_df
