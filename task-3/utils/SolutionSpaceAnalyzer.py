import os
from typing import Any, Callable, Dict, Optional


class SolutionSpaceAnalyzer:
    """
    Generic Solution Space Analyzer that takes analysis function and arguments explicitly.
    """

    def __init__(
        self,
        analysis_func: Callable,
        analysis_args: Optional[Dict[str, Any]] = None,
        analysis_dir: str = "./results/analysis",
        analysis_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the solution space analyzer.

        Args:
            analysis_func: The analysis function to execute
            analysis_args: Dictionary of arguments to pass to the analysis function.
                          If None, only analysis_dir will be passed.
            analysis_dir: Directory to save analysis results
            analysis_name: Optional name for the analysis (for logging)
        """
        self.analysis_func = analysis_func
        self.analysis_args = analysis_args or {}
        self.analysis_dir = analysis_dir
        self.analysis_name = analysis_name or getattr(
            analysis_func, "__name__", "UnnamedAnalysis"
        )

    def run_pipeline(self) -> Any:
        """
        Run the analysis pipeline.

        Returns:
            The result of the analysis function
        """
        print(f"--- Initializing Analysis: {self.analysis_name} ---")

        if not callable(self.analysis_func):
            raise ValueError(f"Analysis function must be callable")
        os.makedirs(self.analysis_dir, exist_ok=True)
        if not self.analysis_args:
            print(f"Running analysis with analysis_dir: {self.analysis_dir}")
            result = self.analysis_func(self.analysis_dir)
        else:
            print(f"Running analysis with arguments: {self.analysis_args}")
            result = self.analysis_func(
                analysis_dir=self.analysis_dir, **self.analysis_args
            )

        print(f"--- Analysis Complete: {self.analysis_name} ---\n")
        return result
