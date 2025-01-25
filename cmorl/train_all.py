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
    cmd = f"python -m cmorl.hyper_search {env} --experiment_name {env}_APS_lambda --steps_per_epoch 2000 --epochs 250 --num_seeds 10"
    os.system(cmd)