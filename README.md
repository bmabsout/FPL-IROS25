# CLOSING THE INTENT-TO-BEHAVIOR GAP VIA FULFILLMENT PRIORITY LOGIC

[ant_walking_resized.webm](https://github.com/user-attachments/assets/2db9e7fd-258b-430d-9288-10cb726d4a48)

## We can learn controllers such as for Hopper-v4 in 7000 steps! 

[Screencast from 2024-09-12 19-47-55.webm](https://github.com/user-attachments/assets/a1de7b93-a8a3-449d-bc66-fe37cecd483d)

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{mabsout2025closing,
  title={Closing the intent-to-behavior gap via Fulfillment Priority Logic},
  author={Mabsout, Bassel El and AbdelGawad, Abdelrahman and Mancuso, Renato},
  journal={arXiv preprint arXiv:2503.05818},
  year={2025}
}
```

## Overview

This repository implements Fulfillment Priority Logic (FPL), a framework for multi-objective reinforcement learning that bridges the gap between human-intuitive behavioral specifications and reinforcement learning optimization. FPL uses power mean operators to compose objectives while preserving logical relationships and priorities.

The main contributions include:
- **Fulfillment Priority Logic (FPL)**: A domain-specific logic for expressing priority-aware objective composition
- **Balanced Policy Gradient (BPG)**: An efficient algorithm that extends DDPG to optimize policies using FPL specifications
- **Sample efficiency improvements**: Up to 500% better sample efficiency compared to Soft Actor Critic

# Installation 

## Via nix
* Install `nix` from https://determinate.systems/posts/determinate-nix-installer
* Run `nix develop --impure` in the repo's root, this should take some time but then you will be dropped in a bash shell with everything required to run training

## Via Conda
* `conda env create --file environment.yml`
* `conda activate cmorl_env`

# Run training
`python envs/Pendulum/train_pendulum.py` - this uses BPG (Balanced Policy Gradient) to train the algorithm and produces automatically checkpoints and logs in the trained folder

`python envs/Pendulum/test_pendulum.py -lr training` - test the trained policy

# Visualization of power mean operators
https://www.math3d.org/RkLQMt2Bl
