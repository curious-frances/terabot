import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def load_training_data(logdir):
    """Load training data from log directory."""
    log_file = os.path.join(logdir, "log.txt")
    if not os.path.exists(log_file):
        raise ValueError(f"No log.txt file found in {log_file}")
    
    print(f"Loading data from: {log_file}")
    
    # Initialize lists to store metrics
    time = []
    iterations = []
    avg_rewards = []
    std_rewards = []
    max_rewards = []
    min_rewards = []
    timesteps = []
    update_norms = []
    learning_rates = []
    delta_stds = []
    deltas_used = []
    num_workers = []
    success_rates = []
    improvements = []
    
    # Read the log file
    with open(log_file, 'r') as f:
        # Skip header line
        next(f)
        for line in f:
            # Parse the line to extract metrics
            parts = line.strip().split('\t')
            time.append(float(parts[0]))
            iterations.append(int(parts[1]))
            avg_rewards.append(float(parts[2]))
            std_rewards.append(float(parts[3]))
            max_rewards.append(float(parts[4]))
            min_rewards.append(float(parts[5]))
            timesteps.append(int(parts[6]))
            update_norms.append(float(parts[7]))
            learning_rates.append(float(parts[8]))
            delta_stds.append(float(parts[9]))
            deltas_used.append(int(parts[10]))
            num_workers.append(int(parts[11]))
            success_rates.append(float(parts[12]))
            improvements.append(float(parts[13]))
    
    return {
        'time': time,
        'iterations': iterations,
        'AverageReward': avg_rewards,
        'StdRewards': std_rewards,
        'MaxRewardRollout': max_rewards,
        'MinRewardRollout': min_rewards,
        'timesteps': timesteps,
        'UpdateNorm': update_norms,
        'LearningRate': learning_rates,
        'DeltaStd': delta_stds,
        'DeltasUsed': deltas_used,
        'NumWorkers': num_workers,
        'SuccessRate': success_rates,
        'Improvement': improvements
    }

def plot_training_metrics(logdir, save_dir=None):
    """Plot various training metrics from ARS training."""
    data = load_training_data(logdir)
    
    # Create separate plots for each metric
    metrics = {
        'Average Reward': ('AverageReward', 'b-'),
        'Max Reward': ('MaxRewardRollout', 'r-'),
        'Min Reward': ('MinRewardRollout', 'g-'),
        'Standard Deviation': ('StdRewards', 'k-'),
        'Timesteps': ('timesteps', 'm-'),
        'Training Time': ('time', 'c-'),
        'Update Norm': ('UpdateNorm', 'y-'),
        'Learning Rate': ('LearningRate', 'purple'),
        'Delta Std': ('DeltaStd', 'orange'),
        'Deltas Used': ('DeltasUsed', 'brown'),
        'Success Rate': ('SuccessRate', 'pink'),
        'Improvement': ('Improvement', 'gray')
    }
    
    for title, (metric, style) in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(data['iterations'], data[metric], style, label=title)
        plt.xlabel('Iteration')
        plt.ylabel(title)
        plt.title(f'{title} Over Training')
        plt.grid(True)
        plt.legend()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}.png'))
        else:
            plt.show()
        plt.close()

def main():
    # Get the log directory from environment variable or use the correct data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get directory of this script
    parent_dir = os.path.dirname(script_dir)  # Get parent directory (main puppersim)
    logdir = os.getenv('ARS_LOG_DIR', os.path.join(parent_dir, 'data'))  # Use data directory in main puppersim
    save_dir = os.getenv('PLOT_SAVE_DIR', os.path.join(parent_dir, 'plots'))
    
    try:
        plot_training_metrics(logdir, save_dir)
        print(f"Plots have been saved to {save_dir}")
    except Exception as e:
        print(f"Error plotting metrics: {str(e)}")
        print("\nAvailable files in log directory:")
        for f in glob.glob(os.path.join(logdir, "*")):
            print(f"  - {f}")

if __name__ == "__main__":
    main() 