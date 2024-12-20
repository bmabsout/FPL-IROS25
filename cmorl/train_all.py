# import subprocess

# # List of environments to run
# environments = ["Reacher-v4", "Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Pendulum-v1", "LunarLanderContinuous-v2"]


# # Common arguments for all environments
# common_args = [
#     "--steps_per_epoch", "2000",
#     "--epochs", "2",
#     "--start_steps", "10000",
#     "--max_ep_len", "1000",
#     "--num_seeds", "10"
# ]

# def run_hypersearch(env_name):
#     # Construct command with environment-specific experiment name
#     cmd = [
#         "python",
#         "-m",
#         "cmorl.hyper_search",
#         env_name,
#         "--experiment_name", f"{env_name}_AQS"
#     ] + common_args
    
#     print(f"\nStarting hyperparameter search for environment: {env_name}")
#     print(f"Command: {' '.join(cmd)}")
    
#     try:
#         # Run the command and capture output
#         process = subprocess.Popen(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             universal_newlines=True
#         )
        
#         # Print output in real-time
#         while True:
#             output = process.stdout.readline()
#             if output == '' and process.poll() is not None:
#                 break
#             if output:
#                 print(output.strip())
                
#         # Wait for the process to complete
#         returncode = process.wait()
        
#         if returncode != 0:
#             print(f"Error running hypersearch for {env_name}. Return code: {returncode}")
#             # Print any error output
#             print(process.stderr.read())
#     except Exception as e:
#         print(f"Exception occurred while running hypersearch for {env_name}: {e}")

# def main():
#     print("Starting hyperparameter search across multiple environments")
#     print(f"Environments to process: {', '.join(environments)}")
    
#     for env_name in environments:
#         run_hypersearch(env_name)
#         print(f"Completed hyperparameter search for {env_name}\n")
    
#     print("All environments processed successfully!")

# if __name__ == "__main__":
#     main()

import os

environments = ["Reacher-v4", "Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Pendulum-v1", "LunarLanderContinuous-v2"]

for env in environments:
    cmd = f"python -m cmorl.hyper_search {env} --experiment_name {env}_AQS --steps_per_epoch 2000 --epochs 2 --start_steps 10000 --num_seeds 10"
    os.system(cmd)