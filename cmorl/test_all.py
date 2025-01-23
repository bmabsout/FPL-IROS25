import argparse
import subprocess
import multiprocessing
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run CMORL tests in parallel")
    parser.add_argument(
        "-n",
        "--num_tests",
        type=int,
        default=10,
        help="Number of test runs per environment",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="test_results",
        help="Directory to store test results",
    )
    return parser.parse_args()


def run_test_command(env_name, env_key_value, num_tests, output_dir):
    """Run the test command for a specific environment"""
    output_path = Path(output_dir) / f"{env_name}.pkl"
    command = [
        "python",
        "-m",
        "cmorl.test",
        "-n",
        str(num_tests),
        "-s",
        str(output_path),
        f"cmorl/trained/{env_name}/{env_name}_AQS/{env_key_value}/seeds/*/epochs/*/",
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully completed testing for {env_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running tests for {env_name}: {e}")


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Environment configurations
    environments = {
        "Reacher-v4": "e:250,start_steps:10000,x:3BTX3L5PYFMGMMF",
        "Hopper-v4": "e:250,start_steps:10000,x:VR2IUVNSYGAS7TQ",
        "Walker2d-v4": "e:250,start_steps:10000,x:ZOMUHZIJRQR5VER",
        "HalfCheetah-v4": "e:250,start_steps:10000,x:UYJJJWLKSGNJKBT",
        "Pendulum-v1": "start_steps:10000,x:HQNBD64SW33TFDK",
        "LunarLanderContinuous-v2": "start_steps:10000,x:JJBGZQGYEUOS3I3",
    }

    # Process each environment sequentially
    for env_name, env_key_value in environments.items():
        print(f"\nStarting tests for {env_name}")
        print(f"Using {args.processes} parallel processes")

        # The cmorl.test module handles parallelization internally
        run_test_command(env_name, env_key_value, args.num_tests, args.output_dir)


if __name__ == "__main__":
    main()
