import time
import os
from typing import Any, Dict, Callable, List, Optional, Tuple


class ExperimentRunner:
    """
    Generic experiment runner class that takes functions and arguments explicitly.
    """

    def __init__(
        self,
        algorithm_name: str,
        algorithm_func: Callable,
        algorithm_args: Dict[str, Any],
        time_file: Optional[str] = "time.txt",
        output_dir: str = "./results/algorithms",
        plot_funcs: Optional[List[Tuple[str, Callable]]] = None,
    ) -> None:
        """
        Initialize the experiment runner.

        Args:
            algorithm_name: Name of the algorithm for logging
            algorithm_func: The algorithm function to execute
            algorithm_args: Dictionary of arguments to pass to the algorithm
            time_file: Path to save execution times (None to disable)
            output_dir: Directory to save results
            plot_funcs: List of (name, function) tuples for plotting
        """
        self.algorithm_name = algorithm_name
        self.algorithm_func = algorithm_func
        self.algorithm_args = algorithm_args
        self.time_file = time_file
        self.output_dir = output_dir
        self.plot_funcs = plot_funcs or []

    def _save_time(self, execution_time_ms: float) -> None:
        """Saves the execution time to the specified file."""
        if self.time_file:
            log_line = f"{self.algorithm_name}: {execution_time_ms:.3f} ms\n"
            try:
                time_file_dir = os.path.dirname(self.time_file)
                if time_file_dir and not os.path.exists(time_file_dir):
                    os.makedirs(time_file_dir)
                with open(self.time_file, "a") as f:
                    f.write(log_line)
            except Exception as e:
                print(f"Error saving execution time to file: {e}")

    def run_pipeline(self) -> Tuple[Any, Any, Any]:
        """
        Run the algorithm pipeline.

        Returns:
            Tuple of (best_ever, final_population, history)
        """
        print(f"--- Initializing Pipeline: {self.algorithm_name} ---")
        start_time = time.time()
        best_ever, final_population, history = self.algorithm_func(
            **self.algorithm_args
        )
        end_time = time.time()

        execution_time_ms = (end_time - start_time) * 1000
        print(f"Execution time: {execution_time_ms:.3f} ms")
        self._save_time(execution_time_ms)

        # Run plotting functions if any
        if self.plot_funcs:
            os.makedirs(self.output_dir, exist_ok=True)

        for plot_name, plot_func in self.plot_funcs:
            if callable(plot_func):
                plot_save_file = f"{self.algorithm_name} -- {plot_name}"
                try:
                    plot_func(
                        best_ever=best_ever,
                        final_population=final_population,
                        history=history,
                        save_dir=self.output_dir,
                        save_filename=plot_save_file,
                    )
                    print(f"Plot '{plot_name}' saved successfully")
                except Exception as e:
                    print(f"Error creating plot '{plot_name}': {e}")
            else:
                print(f"Warning: Plotting function '{plot_name}' is not callable.")

        print(f"--- Pipeline Complete: {self.algorithm_name} ---\n")
        return best_ever, final_population, history
