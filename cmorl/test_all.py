import subprocess

# Environment configurations
environments = [
    "Hopper-v4",
    "Pendulum-v1",
    "LunarLanderContinuous-v2",
    "HalfCheetah-v4",
    "Reacher-v4",
    "Walker2d-v4",
]

for env in environments:
    cmd = f"python -m cmorl.test -n 10 -s {env}.pkl trained/{env}/{env}_APS_lambda/e\:250\,x\:JBAFTCKXO6F6PK2/seeds/*/epochs/*/"
    # cmd = f"python -m cmorl.test -n 10 -s {env}.pkl trained/{env}/{env}_AQS/*/seeds/*/epochs/*/"
    print(cmd)
    subprocess.run(cmd, shell=True)