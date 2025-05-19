# PupperSim Run Commands

This document contains all the commands needed to run training, visualization, and policy execution for both ARS and PPO algorithms.

## Training

### ARS Training
```bash
python puppersim/train.py --algorithm ars --rollout_length 200
```

Key ARS parameters:
- `--rollout_length`: Number of steps per rollout (default: 200)
- `--n_iter`: Number of training iterations (default: 1000)
- `--n_directions`: Number of random directions to explore (default: 16)
- `--n_workers`: Number of parallel workers (default: 32)
- `--step_size`: Learning rate (default: 0.03)
- `--delta_std`: Standard deviation of exploration noise (default: 0.03)

### PPO Training
```bash
python puppersim/train.py --algorithm ppo --rollout_length 200
```

Key PPO parameters:
- `--rollout_length`: Number of steps per rollout (default: 200)
- `--n_iter`: Number of training iterations (default: 1000)
- `--n_workers`: Number of parallel workers (default: 32)
- `--batch_size`: Batch size for updates (default: 64)
- `--n_epochs`: Number of epochs per update (default: 10)
- `--learning_rate`: Learning rate (default: 0.0003)
- `--clip_ratio`: PPO clip ratio (default: 0.2)
- `--entropy_coef`: Entropy coefficient (default: 0.01)
- `--value_coef`: Value function coefficient (default: 0.5)

## Visualization

### Plotting Training Results
```bash
python puppersim/plot_result.py
```

This will:
1. List all available training runs
2. Allow you to select which run to plot
3. Generate separate plots for each metric:
   - Average Reward
   - Max Reward
   - Min Reward
   - Standard Deviation
   - Timesteps
   - Training Time
   - Update Norm
   - Learning Rate
   - Delta Std
   - Deltas Used
   - Success Rate
   - Improvement

Plots are saved in the `plots` directory, organized by run name.

## Running Trained Policies

### ARS Policy
```bash
python puppersim/pupper_ars_run_policy.py --expert_policy_file data/lin_policy_plus_latest.npz --json_file data/params.json --render
```

### PPO Policy
```bash
python puppersim/pupper_ppo_run_policy.py --expert_policy_file data/ppo_policy_latest.pt --json_file data/params.json --render
```

Common parameters for both:
- `--expert_policy_file`: Path to the trained policy file
- `--json_file`: Path to the parameters JSON file
- `--render`: Enable visualization
- `--num_steps`: Number of steps to run (default: 1000)

## File Locations

### Training Data
- Training logs: `data/{algorithm}_{timestamp}/log.txt`
- Policy files: 
  - ARS: `data/lin_policy_plus_latest.npz`
  - PPO: `data/ppo_policy_latest.pt`
- Parameters: `data/params.json`

### Plots
- All plots are saved in: `plots/{algorithm}_{timestamp}/`
- Each metric gets its own plot file

## Notes

- Training logs are saved in timestamped directories to prevent overwriting
- Each training run creates a unique directory with format `{algorithm}_{timestamp}/`
- The environment uses PyBullet for physics simulation
- Both ARS and PPO implementations support parallel training using Ray 