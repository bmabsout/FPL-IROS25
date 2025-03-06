from cmorl.utils.loss_composition import p_mean
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import typer
import keras
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
import os
import multiprocessing
from functools import partial
from tqdm import tqdm

app = typer.Typer()

def plot_values(values, ax=None, title=None):
    """
    Plot compete_x and compete_y values.
    
    Args:
        values: Array of shape (2, steps, samples) containing compete_x and compete_y values
        ax: Optional matplotlib axis to plot on. If None, uses current axis
        title: Optional title for the plot
    """
    if ax is None:
        ax = plt.gca()
    
    # Plot all x values in blue with transparency
    for i in range(values.shape[2]):
        ax.plot(range(values.shape[1]), values[0, :, i], color=to_rgba('C0', 0.03), linewidth=2)
    
    # Plot all y values in orange with transparency
    for i in range(values.shape[2]):
        ax.plot(range(values.shape[1]), values[1, :, i], color=to_rgba('C1', 0.03), linewidth=2)
    
    # Add representative lines for the legend
    ax.plot([], [], color='C0', linewidth=2, label="compete_x")
    ax.plot([], [], color='C1', linewidth=2, label="compete_y")
    
    ax.set_xlabel('Optimization Steps')
    ax.set_ylabel('Value')
    if title:
        ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def run_optimization(
    train_steps: int,
    learning_rate: float,
    random_noise: float,
    p: float,
    num_samples: int
):
    """
    Run the optimization process with the given parameters.
    
    Args:
        train_steps: Number of training steps
        learning_rate: Learning rate for optimization
        random_noise: Amount of random noise to add
        p: p value for p_mean
        num_samples: Number of samples to use
        
    Returns:
        Array of shape (2, train_steps, num_samples) containing compete_x and compete_y values
    """
    # Set random seed based on p to ensure reproducibility but different for each p
    # np.random.seed(int(abs(p * 1000)) % 10000)
    # tf.random.set_seed(int(abs(p * 1000)) % 10000)
    
    np.random.seed(0)
    tf.random.set_seed(0)

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    inits = np.linspace(0.1, 0.499, num_samples)
    x = tf.Variable(inits)
    y = tf.Variable(1-inits)
    values = np.zeros((2, train_steps, num_samples))
    
    for i in range(train_steps):
        with tf.GradientTape() as tape:
            compete_x = (x*(1-y))
            compete_y = (y*(1-x))
            loss = -p_mean([compete_x, compete_y], p=p, axis=0)
        gradients = tape.gradient(loss, [x, y])
        optimizer.apply_gradients(zip(gradients, [x, y]))
        values[0, i] = compete_x
        values[1, i] = compete_y

        x.assign_add(np.random.randn(num_samples)*random_noise)
        y.assign_add(np.random.randn(num_samples)*random_noise)
        x.assign(tf.clip_by_value(x, 0.1, 0.9))
        y.assign(tf.clip_by_value(y, 0.1, 0.9))
    
    return p, values

@app.command()
def optimize(
    train_steps: int = typer.Option(100, help="Number of training steps"),
    learning_rate: float = typer.Option(0.02, help="Learning rate for optimization"),
    random_noise: float = typer.Option(0.0, help="Amount of random noise to add"),
    p: float = typer.Option(0.0, help="p value for p_mean"),
    num_samples: int = typer.Option(500, help="Number of samples to use"),
):
    """Run a single optimization with the specified p-value and plot the results."""
    _, values = run_optimization(
        train_steps=train_steps,
        learning_rate=learning_rate,
        random_noise=random_noise,
        p=p,
        num_samples=num_samples
    )
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plot_values(values, title=f'P-Mean Optimization with p = {p:.2f}')
    plt.show()
    
    return values

def _run_optimization_wrapper(p, train_steps, learning_rate, random_noise, num_samples):
    """Wrapper function for parallel processing"""
    return run_optimization(train_steps, learning_rate, random_noise, p, num_samples)

def create_frame(result_item, temp_dir):
    """
    Create and save a single frame.
    
    Args:
        result_item: Tuple of (p_value, values)
        temp_dir: Directory to save the frame
        
    Returns:
        Path to the saved frame
    """
    idx, (p, values) = result_item
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_values(values, ax=ax, title=f'P-Mean Optimization with p = {p:.2f}')
    frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")
    fig.savefig(frame_path, dpi=100)
    plt.close(fig)
    return frame_path

@app.command()
def animate(
    train_steps: int = typer.Option(300, help="Number of training steps"),
    learning_rate: float = typer.Option(5e-3, help="Learning rate for optimization"),
    random_noise: float = typer.Option(5e-3, help="Amount of random noise to add"),
    num_samples: int = typer.Option(1000, help="Number of samples to use"),
    num_p_values: int = typer.Option(200, help="Number of p-values to test"),
    p_min: float = typer.Option(-5.0, help="Minimum p-value"),
    p_max: float = typer.Option(5.0, help="Maximum p-value"),
    fps: int = typer.Option(10, help="Frames per second in the output video"),
    output_file: str = typer.Option("p_mean_animation.mp4", help="Output video file path"),
    num_processes: int = typer.Option(None, help="Number of parallel processes (default: number of CPU cores)"),
    temp_dir: str = typer.Option("temp_frames", help="Directory to store temporary frame images"),
):
    """Create an animation showing optimization with different p-values."""
    # Create p-values to test
    ps = np.linspace(p_min, p_max, num_p_values)
    
    # Set number of processes
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print(f"Running {num_p_values} optimizations using {num_processes} processes...")
    
    # Run optimizations in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create a partial function with fixed parameters
        func = partial(
            _run_optimization_wrapper,
            train_steps=train_steps,
            learning_rate=learning_rate,
            random_noise=random_noise,
            num_samples=num_samples
        )
        
        # Run the optimizations in parallel and collect results
        results = list(tqdm(pool.imap(func, ps), total=len(ps)))
    
    # Sort results by p-value to ensure correct order
    results.sort(key=lambda x: x[0])
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Creating frames in parallel...")
    
    # Create frames in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create a list of (index, result) tuples
        indexed_results = list(enumerate(results))
        # Use partial to pass the temp_dir parameter
        frame_func = partial(create_frame, temp_dir=temp_dir)
        # Process the frames in parallel
        frame_paths = list(tqdm(pool.imap(frame_func, indexed_results), total=len(results)))
    
    print("Combining frames into video...")
    
    # Use FFmpeg to combine frames into video
    frame_pattern = os.path.join(temp_dir, "frame_%04d.png")
    os.system(f"ffmpeg -y -framerate {fps} -i {frame_pattern} -c:v libx264 -pix_fmt yuv420p {output_file}")
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)
    
    print(f"Animation saved to {output_file}")

if __name__ == "__main__":
    app()
