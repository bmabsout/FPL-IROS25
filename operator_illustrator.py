import tensorflow as tf
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from typing import Callable, List, Optional, Union
import argparse
from cmorl.utils.loss_composition import simple_p_mean, curriculum, p_mean, offset

# TODO: Change the naming of the curriculum function to offset or prioritize
# TODO: remove the class structure and make  everything a tf.function


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


def offset_2(o: List[tf.Tensor], slack: float = 0.1, p: float = -2.0) -> tf.Tensor:
    """
    Compute the offset operation on a list of objectives

    Args:
        objectives: List of objective tensors
        slack: Slack parameter for offset (default: 0.1)
        p: Parameter for p-mean composition (default: -2.0)

    Returns:
        tf.Tensor: Result of the offset operation
    """
    x1, x2 = o
    return min((x1 + slack) / (1 + slack), x2)


class RewardOptimizer:
    def __init__(
        self,
        reward_type: str,
        slack: float = 0.1,
        num_variables: int = 4,
        learning_rate: float = 0.01,
        num_steps: int = 1000,
        competitiveness: float = 0.2,
        randomness: float = 0.01,
        initial_values: Optional[List[float]] = None,
    ):
        """
        Initialize the reward optimization problem

        Args:
            num_variables: Number of variables to optimize (default: 4)
            learning_rate: Learning rate for gradient descent (default: 0.01)
            num_steps: Number of optimization steps (default: 1000)
        """
        self.reward_type = reward_type
        self.slack = slack
        self.num_variables = num_variables
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.randomness = randomness
        self.initial_values = initial_values

        # Initialize variables to optimize
        if initial_values is not None:
            if len(initial_values) != num_variables:
                raise ValueError(
                    f"Initial values length {len(initial_values)} does not match num_variables {num_variables}"
                )
            initial_values = np.array(initial_values)
        else:
            initial_values = np.clip(
                a=np.ones(num_variables) * 0.01
                + np.random.rand(num_variables) * randomness,
                a_max=1.0,
                a_min=0.0,
            )

        self.variables = tf.Variable(initial_values, dtype=tf.float32)

        # Create optimizer
        self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

        # Storage for plotting
        self.o_history = []
        self.reward_history = []
        self.competitiveness = competitiveness

    def compute_outputs(self) -> tf.Tensor:
        """
        Compute the output values for each variable

        Returns:
            tf.Tensor: Tensor of output values following the competitive formula
        """
        uncompeting_objectives = self.variables
        others_means = (
            tf.reduce_sum(uncompeting_objectives) - uncompeting_objectives
        ) / (self.num_variables - 1)
        outputs = uncompeting_objectives * (1 - others_means * self.competitiveness)
        return outputs

    def optimize(
        self,
        reward_composer: Callable[[List[tf.Tensor], float], tf.Tensor],
        p_value: float = -2.0,
    ):
        """
        Run the optimization process

        Args:
            reward_composer: Function that takes (outputs, p_value) and returns a reward
            p_value: Parameter for p-mean composition (default: -2.0)
        """
        for step in range(self.num_steps):
            with tf.GradientTape() as tape:
                # Calculate outputs
                outputs = self.compute_outputs()

                reward = -reward_composer(outputs, p=p_value)

            # Calculate and apply gradients
            gradients = tape.gradient(reward, [self.variables])
            print(f"Gradients: {gradients}")
            print(f"Variables: {self.variables}")
            self.variables.assign(tf.clip_by_value(self.variables - self.learning_rate * gradients[0], 0.0, 1.0))

            # self.optimizer.apply_gradients(zip(gradients, [self.variables]))

            # Store values for plotting
            self.o_history.append([float(o) for o in outputs])
            self.reward_history.append(float(-reward))

    def plot_results(self, p_value: float, save_path: Optional[str] = None, minimal: bool = True):
        """
        Plot optimization results with either minimal or detailed style

        Args:
            p_value: P-value used in optimization (for filename)
            save_path: Optional path to save the plot
            minimal: Whether to use minimal style (True) or detailed style (False)
        """
        if minimal:
            self._plot_minimal(p_value, save_path)
        else:
            self._plot_detailed(p_value, save_path)

    def _plot_minimal(self, p_value: float, save_path: Optional[str] = None):
        """
        Plot minimalistic optimization results with zero white space margins

        Args:
            p_value: P-value used in optimization (for filename only)
            save_path: Optional path to save the plot
        """
        # Create figure with precise dimensions
        plt.figure(figsize=(1.5, 1), facecolor="white")
        ax = plt.gca()

        # Plot objective trajectories with thin lines
        for i in range(self.num_variables):
            ax.plot([o[i] for o in self.o_history], linewidth=1.0)

        # Remove all decorative elements
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Remove border lines (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # No grid
        ax.grid(False)

        # Set zero margins
        plt.margins(0, 0)

        # Extend subplot to figure edges completely
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Handle save path creation
        if save_path is None:
            if not os.path.exists("results"):
                os.makedirs("results")
            if not os.path.exists(f"results/{self.reward_type}"):
                os.makedirs(f"results/{self.reward_type}")
            save_path = (
                f"./results/{self.reward_type}/minimal_{self.reward_type}_"
                f"p_{p_value}.pdf"
            )

        # plot a vertical line at the place where o[0] is 0.5
        # find the index where o[0] is closest to 0.5
        closest_index = np.argmin(
            np.abs(np.array(self.o_history)[:, 0] - self.slack / 2)
        )
        print(f"Closest index to {self.slack/2}: {closest_index}")
        print(f"Value at closest index: {self.o_history[closest_index][0]}")
        # plot a vertical line at that index
        if self.reward_type == "curriculum":
            plt.axvline(x=closest_index, color="black", linestyle="--", linewidth=1.0)

        # Set pad_inches=0 to eliminate all padding during save
        plt.savefig(
            save_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=False
        )
        plt.close()

    def _plot_detailed(self, p_value: float, save_path: Optional[str] = None):
        """
        Plot detailed optimization results with proper axes, labels, and grid

        Args:
            p_value: P-value used in optimization (for plot title)
            save_path: Optional path to save the plot
        """
        # Initialize the figure with appropriate dimensions
        plt.figure(figsize=(10, 6))

        # Create main axes for unified plotting
        ax = plt.gca()

        # Plot individual output trajectories
        # Using different line styles for visual distinction
        line_styles = ["-", "--", "-.", ":"]
        for i in range(self.num_variables):
            ax.plot(
                [o[i] for o in self.o_history],
                label=f"Output {i+1}",
                linestyle=line_styles[i % len(line_styles)],
                linewidth=1.5,
            )

        # Plot reward trajectory with distinct appearance
        ax.plot(
            self.reward_history, label="Reward", color="red", linewidth=1.7, alpha=0.5
        )

        # Configure axis labels and limits
        ax.set_xlabel("Gradient Steps")
        ax.set_ylabel("Values")
        
        # Set y-axis from 0 to 1 for objectives
        ax.set_ylim(0, 1)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle="--")

        # Create comprehensive title with experiment parameters
        reward_type_display = self.reward_type.upper()
        title = (
            f"{reward_type_display} Optimization\n"
            f"p={p_value}, slack={self.slack:.2f}, "
            f"competitiveness={self.competitiveness:.2f}"
        )
        plt.title(title)

        # Position legend for optimal visibility
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        # Adjust layout to prevent label clipping
        plt.tight_layout()

        # Handle save path creation
        if save_path is None:
            if not os.path.exists("results"):
                os.makedirs("results")
            if not os.path.exists(f"results/{self.reward_type}"):
                os.makedirs(f"results/{self.reward_type}")
            save_path = (
                f"./results/{self.reward_type}/detailed_{self.reward_type}_"
                f"p_{p_value}.pdf"
            )

        # Plot a vertical line at the place where o[0] is closest to slack/2
        if self.reward_type == "curriculum":
            closest_index = np.argmin(
                np.abs(np.array(self.o_history)[:, 0] - self.slack / 2)
            )
            plt.axvline(x=closest_index, color="black", linestyle="--", linewidth=1.0)

        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()


def main(
    num_variables: int = 4,
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    p_value: float = -2.0,
    slack: float = 0.1,
    reward_type: str = "curriculum",
    competitiveness: float = 0.2,
    randomness: float = 0.01,
    initial_values: Optional[List[float]] = None,
    minimal_plot: bool = True,
):
    """
    Main function to run the optimization experiment

    Args:
        num_variables: Number of variables to optimize (default: 4)
        learning_rate: Learning rate for gradient descent (default: 0.01)
        num_steps: Number of optimization steps (default: 1000)
        p_value: Parameter for p-mean composition (default: -2.0)
        slack: Slack parameter for curriculum composer (default: 0.1)
        reward_type: Type of reward composer to use ["curriculum", "pmean"] (default: "curriculum")
        competitiveness: amount of competitiveness to use in the objectives (default: 0.2)
        randomness: amount of randomness to use in the initial values (default: 0.01)
        minimal_plot: Whether to use minimal plotting style (default: True)
    """
    # Select reward composer based on type
    if reward_type == "curriculum":
        reward_composer = lambda outputs, p: curriculum(outputs, slack=slack, p=p)
    elif reward_type == "pmean":
        reward_composer = lambda outputs, p: simple_p_mean(outputs, p)
    elif reward_type == "AND":
        reward_composer = lambda outputs, p: AND(outputs, p)
    elif reward_type == "OR":
        reward_composer = lambda outputs, p: OR(outputs, p)
    elif reward_type == "min":
        reward_composer = lambda outputs, p: tf.reduce_min(outputs)
    elif reward_type == "max":
        reward_composer = lambda outputs, p: tf.reduce_max(outputs)
    elif reward_type == "offset":
        reward_composer = lambda outputs, p: offset_2(outputs, slack=slack, p=p)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

    # Create optimizer instance
    optimizer = RewardOptimizer(
        reward_type=reward_type,
        slack=slack,
        num_variables=num_variables,
        learning_rate=learning_rate,
        num_steps=num_steps,
        competitiveness=competitiveness,
        randomness=randomness,
        initial_values=initial_values,
    )

    # Run optimization
    optimizer.optimize(reward_composer=reward_composer, p_value=p_value)

    # Plot results
    optimizer.plot_results(p_value=p_value, minimal=minimal_plot)


def parse_float_list(arg):
    """Convert a comma-separated string to a list of floats."""
    if arg is None:
        return None
    return [float(x) for x in arg.split(",")]


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run reward optimization experiment")
    parser.add_argument(
        "--num_variables", type=int, default=4, help="Number of variables to optimize"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent",
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of optimization steps"
    )
    parser.add_argument(
        "--p_value", type=float, default=-2.0, help="Parameter for p-mean composition"
    )
    parser.add_argument(
        "--slack",
        type=float,
        default=0.1,
        help="Slack parameter for curriculum composer",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="curriculum",
        choices=["curriculum", "pmean", "AND", "OR", "min", "max", "offset"],
        help="Type of reward composer to use",
    )
    parser.add_argument(
        "--competitiveness",
        type=float,
        default=0.2,
        help="Competitiveness parameter for curriculum composer",
    )
    parser.add_argument(
        "--randomness",
        type=float,
        default=0.01,
        help="Randomness parameter for curriculum composer",
    )
    parser.add_argument(
        "--initial_values",
        type=parse_float_list,
        default=None,
        help="Initial values for the variables to optimize (comma-separated list of floats)",
    )
    parser.add_argument(
        "--minimal_plot",
        action="store_true",
        default=True,
        help="Use minimal plotting style (default: True)",
    )
    parser.add_argument(
        "--detailed_plot",
        action="store_false",
        dest="minimal_plot",
        help="Use detailed plotting style with axes, grid, and labels",
    )

    args = parser.parse_args()
    np.random.seed(42)
    # Run main function with parsed arguments
    main(
        num_variables=args.num_variables,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        p_value=args.p_value,
        slack=args.slack,
        reward_type=args.reward_type,
        competitiveness=args.competitiveness,
        randomness=args.randomness,
        initial_values=args.initial_values,
        minimal_plot=args.minimal_plot,
    )
