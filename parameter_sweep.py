import itertools
import subprocess
import os
from typing import Dict, List
import concurrent.futures
import tqdm


# In parameter_sweep.py, update the run_experiment function:


def run_experiment(params: Dict[str, any]) -> None:
    """
    Run a single experiment with the given parameters

    Args:
        params: Dictionary of parameter name to parameter value
    """
    cmd = ["python", "operator_illustrator.py"]

    for key, value in params.items():
        # Special handling for initial_values parameter with list values
        if key == "initial_values" and isinstance(value, list):
            # Flatten nested list if present (e.g., [[1.0, 0.0]] to [1.0, 0.0])
            if len(value) == 1 and isinstance(value[0], list):
                value = value[0]

            # Convert the list to a comma-separated string
            value_str = ",".join(str(x) for x in value)

            # Use equals syntax to avoid problems with negative numbers
            cmd.append(f"--{key}={value_str}")

        # Special handling for any negative numeric values
        elif isinstance(value, (int, float)) and value < 0:
            # Use equals syntax for negative values to prevent misinterpretation
            cmd.append(f"--{key}={value}")

        # Standard handling for all other parameters
        else:
            cmd.extend([f"--{key}", str(value)])

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# Define parameter ranges to sweep
parameter_grid = {
    "num_variables": [
        2,
    ],  # Fixed for this experiment
    "learning_rate": [
        0.005,
        # 0.01,
        # 0.05,
        # 0.1,
    ],
    "num_steps": [
        5_000,
    ],  # Fixed for this experiment
    "p_value": [
        # -10.0,
        -2.0,
        # -1.0,
        # 0.0,
        # 0.1,
    ],
    "slack": [
        # 0.3,
        # 0.5,
        # 0.7,
        # 0.9,
        1.0,
        # 1.5,
    ],
    "reward_type": [
        # "AND",
        # "OR",
        "curriculum",
        # "offset"
    ],
    "competitiveness": [
        # 0.0,
        0.2,
        # 0.4,
        # 0.5,  # This is the default value
        # 0.6,
        # 0.8,
        # 1.0,
    ],
    "randomness": [
        # 0.0,
        # 0.1,
        # 0.3,
        1,
    ],
    # "initial_values": [[-1, 2.5]],  # Fixed for AND experiment
    # "initial_values": [[0.5, 0.8]],  # Fixed for OR experiment
    "initial_values": [[-1, -1]],  # Fixed for curriculum experiment
}


def main():
    # Create output directory for results
    os.makedirs("experiment_results", exist_ok=True)

    # Generate all combinations of parameters
    keys, values = zip(*parameter_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total number of experiments to run: {len(combinations)}")

    # Run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all experiments
        futures = []
        for params in combinations:
            future = executor.submit(run_experiment, params)
            futures.append(future)

        # Show progress bar
        for _ in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Running experiments",
        ):
            pass


if __name__ == "__main__":
    main()
