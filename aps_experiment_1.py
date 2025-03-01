import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os
from collections import defaultdict
import keras
from cmorl.utils.loss_composition import curriculum, p_mean


def AND(objectives: List[tf.Tensor], p: float = -2.0) -> tf.Tensor:
    """
    Compute the AND operation on a list of objectives

    Args:
        objectives: List of objective tensors
        p: Parameter for p-mean composition (default: -2.0)

    Returns:
        tf.Tensor: Result of the AND operation
    """
    return p_mean(objectives, p=p)


# define an OR operator which is basically a p_mean_stable with a passed p value as an argument
# also it is basically an AND with Demorgan's law
def OR(objectives: List[tf.Tensor], p: float = -2.0) -> tf.Tensor:
    """
    Compute the OR operation on a list of objectives

    Args:
        objectives: List of objective tensors
        p: Parameter for p-mean composition (default: -2.0)

    Returns:
        tf.Tensor: Result of the OR operation
    """
    objectives = 1 - objectives
    return 1 - AND(objectives, p=p)


class PValueExperiment:
    def __init__(
        self,
        reward_type: str,
        p_values: List[float],
        num_variables: int = 4,
        learning_rate: float = 0.01,
        num_steps: int = 1000,
        slack: float = 0.1,
        competitiveness: float = 0.2,
        randomness: float = 0.01,
        num_runs: int = 5,  # Number of runs per p-value for statistical significance
    ):
        """
        Initialize the p-value analysis experiment

        Args:
            reward_type: Type of reward composer ("AND", "OR", "curriculum")
            p_values: List of p-values to test
            num_variables: Number of variables to optimize
            learning_rate: Learning rate for optimization
            num_steps: Number of steps per optimization run
            slack: Slack parameter for curriculum
            competitiveness: Competitiveness parameter
            randomness: Initial randomness
            num_runs: Number of runs per p-value
        """
        self.reward_type = reward_type
        self.p_values = p_values
        self.num_variables = num_variables
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.slack = slack
        self.competitiveness = competitiveness
        self.randomness = randomness
        self.num_runs = num_runs

        # Storage for results
        self.results = defaultdict(list)

    def run_single_experiment(self, p_value: float) -> List[float]:
        """Run a single optimization experiment with given p-value"""
        # Initialize variables
        initial_values = np.clip(
            a=np.ones(self.num_variables) * 0.01
            + np.random.rand(self.num_variables) * self.randomness,
            a_max=1.0,
            a_min=0.0,
        )
        # sort the initial values so that the first one is the largest
        if self.reward_type != "curriculum":
            initial_values = np.sort(initial_values)[::-1]

        variables = tf.Variable(initial_values, dtype=tf.float32)
        optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)

        # Run optimization
        for _ in range(self.num_steps):
            with tf.GradientTape() as tape:
                # Calculate outputs
                uncompeting_objectives = tf.tanh(tf.abs(variables))
                others_means = (
                    tf.reduce_sum(uncompeting_objectives) - uncompeting_objectives
                ) / (self.num_variables - 1)
                outputs = uncompeting_objectives * (
                    1 - others_means * self.competitiveness
                )

                # Calculate reward based on type
                if self.reward_type == "AND":
                    reward = -AND(outputs, p=p_value)
                elif self.reward_type == "OR":
                    reward = -OR(outputs, p=p_value)
                elif self.reward_type == "curriculum":
                    reward = -curriculum(outputs, slack=self.slack, p=p_value)
                else:
                    raise ValueError(f"Unknown reward type: {self.reward_type}")

            # Apply gradients
            gradients = tape.gradient(reward, [variables])
            optimizer.apply_gradients(zip(gradients, [variables]))

        # Return final objective values
        uncompeting_objectives = tf.tanh(tf.abs(variables))
        others_means = (
            tf.reduce_sum(uncompeting_objectives) - uncompeting_objectives
        ) / (self.num_variables - 1)
        final_outputs = uncompeting_objectives * (
            1 - others_means * self.competitiveness
        )
        return [float(x) for x in final_outputs.numpy()]

    def run_experiments(self):
        """Run experiments for all p-values"""
        for p_value in self.p_values:
            print(f"Running experiments for p={p_value}")
            for run in range(self.num_runs):
                final_objectives = self.run_single_experiment(p_value)
                self.results[p_value].append(final_objectives)

    def plot_results(self):
        """
        Plot the results with p-values on x-axis and final objective values as bars.
        Creates a simple bar chart showing how objectives vary with p-value.
        """
        plt.figure(figsize=(15, 6))

        # For each p-value, calculate the mean final value of each objective
        means_obj1 = []  # First objective means
        means_obj2 = []  # Second objective means

        # Extract means for each p-value
        for p in self.p_values:
            # Get all runs for this p-value
            runs = self.results[p]

            # Calculate mean for each objective
            obj1_values = [run[0] for run in runs]  # First objective values
            obj2_values = [run[1] for run in runs]  # Second objective values

            means_obj1.append(np.mean(obj1_values))
            means_obj2.append(np.mean(obj2_values))

        # Create bar positions
        x = np.arange(len(self.p_values))
        width = 0.35  # Width of bars

        # Plot bars for each objective
        plt.bar(x - width / 2, means_obj1, width, label="Objective 1", color="skyblue")
        plt.bar(
            x + width / 2, means_obj2, width, label="Objective 2", color="lightgreen"
        )

        # Customize plot
        plt.xlabel("p")
        plt.ylabel("Final Objective Value")
        # plt.title(f"{self.reward_type} Operator: Final Objective Values vs p-value")
        # plt.title(f"Final Objective Values vs p-value Operator")
        plt.xticks(x, [str(p) for p in self.p_values], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save plot
        if not os.path.exists("p_value_analysis"):
            os.makedirs("p_value_analysis")
        # plt.savefig(f"p_value_analysis/{self.reward_type}_p_value_analysis.png")
        plt.savefig(f"aps_exp/p_value_analysis/p_value_analysis.pdf", format="pdf")
        plt.close()


# Run experiments for each operator
def run_all_experiments():
    # Define p-values to test from -20 to 20 with a step of 1
    p_values = np.arange(-20, 21, 1).tolist()

    # Parameters for experiments
    params = {
        "num_variables": 2,
        "learning_rate": 0.005,
        "num_steps": 1000,
        "slack": 0.5,
        "competitiveness": 0.2,
        "randomness": 1,
        "num_runs": 10,
    }

    # Run experiments for each operator
    for operator in [
        "AND",
        #   "OR",
        #   "curriculum"
    ]:
        print(f"\nRunning {operator} experiments...")
        experiment = PValueExperiment(operator, p_values, **params)
        experiment.run_experiments()
        experiment.plot_results()


if __name__ == "__main__":
    run_all_experiments()
