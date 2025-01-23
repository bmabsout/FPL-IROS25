import subprocess

# Environment configurations
environments = {
    # "Reacher-v4": "e:250,start_steps:10000,x:3BTX3L5PYFMGMMF",
    "Hopper-v4": "e:250,start_steps:10000,x:VR2IUVNSYGAS7TQ",
    "Walker2d-v4": "e:250,start_steps:10000,x:ZOMUHZIJRQR5VER",
    "HalfCheetah-v4": "e:250,start_steps:10000,x:UYJJJWLKSGNJKBT",
    "Pendulum-v1": "start_steps:10000,x:HQNBD64SW33TFDK",
    "LunarLanderContinuous-v2": "start_steps:10000,x:JJBGZQGYEUOS3I3",
}


for env, config in environments.items():
    cmd = f"python -m cmorl.test -n 10 -s {env}.pkl cmorl/trained/{env}/{env}_AQS/*/seeds/*/epochs/*/"
    # cmd = f"python -m cmorl.test -n 10 -s {env}.pkl trained/{env}/{env}_AQS/*/seeds/*/epochs/*/"
    print(cmd)
    subprocess.run(cmd, shell=True)
