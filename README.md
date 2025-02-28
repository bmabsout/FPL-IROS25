# EXPRESSIVE REINFORCEMENT LEARNING VIA ALGEBRAIC Q-VALUE SCALARIZATION

[ant_walking_resized.webm](https://github.com/user-attachments/assets/2db9e7fd-258b-430d-9288-10cb726d4a48)

## We can learn controllers such as for Hopper-v4 in 7000 steps! 

[Screencast from 2024-09-12 19-47-55.webm](https://github.com/user-attachments/assets/a1de7b93-a8a3-449d-bc66-fe37cecd483d)


# Installation 

## Via nix
* Install `nix` from https://determinate.systems/posts/determinate-nix-installer
* Run `nix develop --impure`` in the repo's root, this should take some time but then you will be dropped in a bash shell with everything required to run training

## Via Conda
* `conda env create --file environment.yml`
* `conda activate cmorl_env`

# Run training
`python envs/Pendulum/train_pendulum.py`, this uses ddpg to train the algorithm and produces automatically checkpoints and logs in the trained folder
`python envs/Pendulum/test_pendulum.py -lr training`


# Visualization of pmean
https://www.math3d.org/RkLQMt2Bl