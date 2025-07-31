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

# Key Concepts

## Fulfillment Priority Logic (FPL)
FPL allows practitioners to define logical formulas representing their intentions and priorities within multi-objective reinforcement learning. It uses power mean operators to smoothly interpolate between minimum (p→-∞) and maximum (p→∞) operations, generalizing linear utilities.

## FPL Syntax
```
ϕ ::= f | ϕ ∧p ϕ | ϕ ∨p ϕ | ¬ϕ | [ϕ]δ
```
where:
- `f ∈ [0, 1]` denotes a base fulfillment value
- `p ≤ 0` in both `∧p` and `∨p` operators controls strictness
- `¬` denotes logical negation
- `[ϕ]δ` offsets the priority of ϕ by δ ∈ [-1, 1]

## Balanced Policy Gradient (BPG)
BPG extends Deep Deterministic Policy Gradient (DDPG) to efficiently optimize policies for Multi-Fulfillment MDPs using FPL specifications. It works with Fulfillment Q-values (FQ-values) that represent the degree to which each objective is fulfilled across time.

# Visualization of power mean operators
https://www.math3d.org/RkLQMt2Bl
