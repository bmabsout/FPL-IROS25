import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from typing import Callable, List, Optional, Union
import argparse
from cmorl.utils.loss_composition import simple_p_mean, then, curriculum, p_mean

# NOTE: Remeber that the OR operator needs preturbation
# TODO: Implement the OR operator using DeMorgan's law
# TODO: Implement the AND operator
# TODO: Change the naming of the curriculum function to offset or prioritize
# TODO: remove the class structure and make everything a tf.function
# TODO: Case 1: Showcase how the OR operator functions on 4 variables
# TODO: Case 2: Showcase how the AND operator functions on 4 variables
# TODO: Case 3: Showcase how the offset operator functions on 4 variables


class RewardOptimizer:
    def __init__(
        self, num_variables: int = 4, learning_rate: float = 0.01, num_steps: int = 1000, competitiveness: float = 0.2, randomness: float = 0.01
    ):
        """
        Initialize the reward optimization problem

        Args:
            num_variables: Number of variables to optimize (default: 4)
            learning_rate: Learning rate for gradient descent (default: 0.01)
            num_steps: Number of optimization steps (default: 1000)
        """
        self.num_variables = num_variables
        self.learning_rate = learning_rate
        self.num_steps = num_steps

        # Initialize variables to optimize
        initial_values = np.ones(num_variables) * 0.01+np.random.rand(num_variables)*randomness
        # initial_values[0] = 0.5
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
        uncompeting_objectives = tf.tanh(tf.abs(self.variables))
        others_means = (tf.reduce_sum(uncompeting_objectives)-uncompeting_objectives)/(self.num_variables-1)
        outputs = uncompeting_objectives * (1 - others_means*self.competitiveness)
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

                # Calculate reward using provided composer
                reward = -reward_composer(outputs, p=p_value)

            # Calculate and apply gradients
            gradients = tape.gradient(reward, [self.variables])
            self.optimizer.apply_gradients(zip(gradients, [self.variables]))

            # Store values for plotting
            self.o_history.append([float(o) for o in outputs])
            self.reward_history.append(float(-reward))

    def plot_results(self, p_value: float, save_path: Optional[str] = None):
        """
        Plot the optimization results

        Args:
            p_value: P-value used in optimization (for plot title)
            save_path: Optional path to save the plot (default: None)
        """
        plt.figure(figsize=(12, 4))

        # Plot outputs
        plt.subplot(1, 2, 1)
        for i in range(self.num_variables):
            plt.plot([o[i] for o in self.o_history], label=f"o{i+1}")
        plt.xlabel("Gradient Steps")
        plt.ylabel("Output Values")
        plt.title("Output Values vs. Gradient Steps")
        plt.legend()
        plt.grid(True)

        # Plot reward
        plt.subplot(1, 2, 2)
        plt.plot(self.reward_history, label="Reward")
        plt.xlabel("Gradient Steps")
        plt.ylabel("Reward Value")
        plt.title("Reward vs. Gradient Steps")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = f"toy_ex_APS_P_mean_{p_value}_lr_{self.learning_rate}_steps_{self.num_steps}.png"

        plt.savefig(save_path)
        plt.show()


def main(
    num_variables: int = 4,
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    p_value: float = -2.0,
    slack: float = 0.1,
    reward_type: str = "curriculum",
    competitiveness: float = 0.2,
    randomness: float = 0.01,
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
    """
    # Create optimizer instance
    optimizer = RewardOptimizer(
        num_variables=num_variables, learning_rate=learning_rate, num_steps=num_steps, competitiveness=competitiveness, randomness=randomness
    )

    # Select reward composer based on type
    if reward_type == "curriculum":
        reward_composer = lambda outputs, p: curriculum(outputs, slack=slack, p=p)
        save_path = f"curriculum_slack_{slack}_p_{p_value}_lr_{learning_rate}_steps_{num_steps}.png"
    elif reward_type == "pmean":
        reward_composer = lambda outputs, p: simple_p_mean(outputs, p)
        save_path = f"pmean_p_{p_value}_lr_{learning_rate}_steps_{num_steps}.png"
    elif reward_type == "pmean_stable":
        reward_composer = lambda outputs, p: p_mean(outputs, p)
        save_path = f"pmean_stable_p_{p_value}_lr_{learning_rate}_steps_{num_steps}.png"
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

    # Run optimization
    optimizer.optimize(reward_composer=reward_composer, p_value=p_value)

    # Plot results
    optimizer.plot_results(p_value=p_value, save_path=save_path)


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
        "--num_steps", type=int, default=1000, help="Number of optimization steps"
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
        choices=["curriculum", "pmean", "pmean_stable"],
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
    )
