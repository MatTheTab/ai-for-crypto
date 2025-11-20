import yaml
from utils.ExperimentRunner import ExperimentRunner
from utils.SolutionSpaceAnalyzer import SolutionSpaceAnalyzer

solution_space_analysis = SolutionSpaceAnalyzer("task-3/run.yaml")
solution_space_analysis.run_pipeline()

with open("run.yaml", "r") as f:
    config = yaml.safe_load(f)
algorithms = list(config.get("algorithms", {}).keys())

for algorithm in algorithms:
    print(f"Running Algorithm: {algorithm}")
    experiment_runner = ExperimentRunner(
        algorithm_name="test_algorithm", config_path="task-3/run.yaml"
    )
    experiment_runner.run_pipeline()
