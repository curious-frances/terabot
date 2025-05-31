# PupperSim

This repository contains the simulation and training code for the Pupper robot.

## Training

### ARS Training
To train using ARS (Augmented Random Search):
```bash
python puppersim/train.py --algorithm ars --rollout_length 200
```

### PPO Training
To train using PPO (Proximal Policy Optimization):
```bash
python puppersim/train.py --algorithm ppo --rollout_length 200
```

## Running Policies in Simulation

### ARS Policy
To run an ARS policy in simulation:
```bash
python puppersim/pupper_ars_run_policy.py --expert_policy_file data/ars_<timestamp>/lin_policy_plus_latest.npz --json_file data/params.json --render --realtime
```

### PPO Policy
To run a PPO policy in simulation:
```bash
python puppersim/pupper_ppo_run_policy.py --expert_policy_file data/ppo_<timestamp>/policy_plus_latest.npz --json_file data/params.json --render --realtime
```

## Deploying to Robot

### ARS Policy Deployment
To deploy an ARS policy to the physical robot:
```bash
python puppersim/pupper_ars_run_policy.py --expert_policy_file data/ars_<timestamp>/lin_policy_plus_latest.npz --json_file data/params.json --run_on_robot
```

### PPO Policy Deployment
To deploy a PPO policy to the physical robot:
```bash
python puppersim/pupper_ppo_run_policy.py --expert_policy_file data/ppo_<timestamp>/policy_plus_latest.npz --json_file data/params.json --run_on_robot
```

## Notes
- Replace `<timestamp>` with the actual timestamp of your training run
- The `--realtime` flag is important for simulation to run at real-time speed
- The `--render` flag enables visualization in simulation
- The `--run_on_robot` flag is used for deployment to the physical robot
- Make sure the robot is powered on and connected before deployment
- The robot's motors should be calibrated before running any policy

## Troubleshooting
If you encounter the error `AttributeError: 'dict' object has no attribute 'env_specs'`, make sure you're using the correct command for your environment:
- For simulation: Use the commands with `--render --realtime`
- For robot deployment: Use the commands with `--run_on_robot` 