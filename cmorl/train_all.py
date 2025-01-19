import os

environments = [
    "Reacher-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "HalfCheetah-v4",
    "Pendulum-v1",
    "LunarLanderContinuous-v2",
]

for env in environments:
    cmd = f"python -m cmorl.hyper_search {env} --experiment_name {env}_AQS --steps_per_epoch 2000 --epochs 2 --start_steps 1000 --num_seeds 1"
    os.system(cmd)
