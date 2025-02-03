import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from typing import Callable, List, Optional, Union
import argparse

# NOTE: Remeber that the OR operator needs preturbation
# TODO: Implement the OR operator using DeMorgan's law
# TODO: Implement the AND operator
# TODO: Change the naming of the curriculum function to offset or prioritize
# TODO: remove the class structure and make everything a tf.function
# TODO: Case 1: Showcase how the OR operator functions on 4 variables
# TODO: Case 2: Showcase how the AND operator functions on 4 variables
# TODO: Case 3: Showcase how the offset operator functions on 4 variables


@tf.function
def tf_pop(tensor, axis):
    return tf.concat(
        [tf.slice(tensor, [0], [axis]), tf.slice(tensor, [axis + 1], [-1])], 0
    )


@tf.function
def clip_preserve_grads(val, min, max):
    clip_t = tf.clip_by_value(val, min, max)
    return val + tf.stop_gradient(clip_t - val)


@tf.function
def p_mean_stable(
    l: tf.Tensor, p: float, slack=1e-7, default_val=0.0, axis=None, dtype=None
) -> tf.Tensor:
    """
    The Generalized mean
    l: a tensor of elements we would like to compute the p_mean with respect to, elements must be > 0.0
    p: the value of the generalized mean, p = -1 is the harmonic mean, p = 1 is the regular mean, p=inf is the max function ...
    slack: allows elements to be at 0.0 with p < 0.0 without collapsing the pmean to 0.0 fully allowing useful gradient information to leak
    axis: axis or axese to collapse the pmean with respect to, None would collapse all
    https://www.wolframcloud.com/obj/26a59837-536e-4e9e-8ed1-b1f7e6b58377
    """
    l = tf.convert_to_tensor(l)
    dtype = dtype if dtype else l.dtype
    slack = tf.cast(slack, dtype)
    l = tf.cast(l, dtype)
    slacked = l + slack
    p = tf.cast(p, dtype)
    default_val = tf.cast(default_val, dtype)
    min_val = tf.constant(1e-5, dtype=dtype)
    p = tf.where(tf.abs(p) < min_val, -min_val if p < 0.0 else min_val, p)

    stabilizer = tf.reduce_min(slacked) if p < 1.0 else tf.reduce_max(slacked)
    stabilized_l = (
        slacked / stabilizer
    )  # stabilize the values to prevent overflow or underflow

    p_meaned = (
        tf.cond(
            tf.reduce_prod(tf.shape(slacked))
            == 0,  # condition if an empty array is fed in
            lambda: (
                tf.broadcast_to(default_val, tf_pop(tf.shape(slacked), axis))
                if axis
                else default_val
            ),
            lambda: (tf.reduce_mean(stabilized_l**p, axis=axis)) ** (1.0 / p) - slack,
        )
        * stabilizer
    )

    return clip_preserve_grads(p_meaned, tf.reduce_min(l), tf.reduce_max(l))


@tf.function
def p_mean(values, p):
    """
    Calculate the generalized p-mean of a list of values.
    For p → -∞: min(values)
    For p = -1: harmonic mean
    For p → 0: geometric mean
    For p = 1: arithmetic mean
    For p → ∞: max(values)

    Args:
        values: List of values to compute p-mean over
        p: Power parameter determining the type of mean

    Returns:
        tf.Tensor: The p-mean of the input values
    """
    values = tf.stack(values)
    if tf.abs(p) < 1e-10:  # p ≈ 0
        return tf.exp(tf.reduce_mean(tf.math.log(values)))
    else:
        return tf.pow(tf.reduce_mean(tf.pow(values, p)), 1 / p)


@tf.function
def then(x, y, slack=0.5, p=-1.0):
    """
    Implements a priority operator: optimize x then y
    Uses p-mean to create a continuous approximation of logical implication

    Args:
        x: First value to optimize
        y: Second value to optimize
        slack: Slack parameter for smoothing (default: 0.5)
        p: Power parameter for p-mean (default: -1.0)

    Returns:
        tf.Tensor: Priority composition of x and y
    """
    slack = tf.cast(slack, x.dtype)
    min_p_mean = p_mean([0, slack], p=p)
    return (p_mean([x, slack + y * (1 - slack)], p=p) - min_p_mean) / (1.0 - min_p_mean)


@tf.function
def curriculum(l: List[tf.Tensor], slack: float = 0.1, p: float = 0.0) -> tf.Tensor:
    """
    Creates a curriculum of objectives using the then operator

    Args:
        l: List of objectives to compose
        slack: Slack parameter for then operator (default: 0.1)
        p: Power parameter for p-mean (default: 0.0)

    Returns:
        tf.Tensor: Composed curriculum objective
    """
    return reduce(lambda x, y: then(y, x, slack=slack, p=p), reversed(l))


class RewardOptimizer:
    def __init__(
        self, num_variables: int = 4, learning_rate: float = 0.01, num_steps: int = 1000
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
        initial_values = np.ones(num_variables) * 0.01
        initial_values[0] = 0.5
        self.variables = tf.Variable(initial_values, dtype=tf.float32)

        # Create optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        # Storage for plotting
        self.o_history = []
        self.reward_history = []

    def compute_outputs(self) -> List[tf.Tensor]:
        """
        Compute the output values for each variable

        Returns:
            List[tf.Tensor]: List of output values following the competitive formula
        """
        outputs = []
        for i in range(self.num_variables):
            # Calculate mean of other variables
            others = tf.concat([self.variables[:i], self.variables[i + 1 :]], axis=0)
            others_mean = tf.reduce_mean(tf.tanh(tf.abs(others)))

            # Calculate output with competitive term
            output = 0.2 + 0.8 * tf.tanh(tf.abs(self.variables[i])) - 0.2 * others_mean
            outputs.append(output)
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
    """
    # Create optimizer instance
    optimizer = RewardOptimizer(
        num_variables=num_variables, learning_rate=learning_rate, num_steps=num_steps
    )

    # Select reward composer based on type
    if reward_type == "curriculum":
        reward_composer = lambda outputs, p: curriculum(outputs, slack=slack, p=p)
        save_path = f"curriculum_slack_{slack}_p_{p_value}_lr_{learning_rate}_steps_{num_steps}.png"
    elif reward_type == "pmean":
        reward_composer = lambda outputs, p: p_mean(outputs, p)
        save_path = f"pmean_p_{p_value}_lr_{learning_rate}_steps_{num_steps}.png"
    elif reward_type == "pmean_stable":
        reward_composer = lambda outputs, p: p_mean_stable(outputs, p)
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

    args = parser.parse_args()

    # Run main function with parsed arguments
    main(
        num_variables=args.num_variables,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        p_value=args.p_value,
        slack=args.slack,
        reward_type=args.reward_type,
    )
