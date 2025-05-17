import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
from pathlib import Path

def load_training_data(logdir):
    """Load training data from log directory."""
    # Find all log files
    log_files = glob.glob(os.path.join(logdir, "*.json"))
    if not log_files:
        raise ValueError(f"No log files found in {logdir}")
    
    # Load the most recent log file
    latest_log = max(log_files, key=os.path.getctime)
    
    with open(latest_log, 'r') as f:
        data = json.load(f)
    
    return data

def plot_training_metrics(logdir, save_dir=None):
    """Plot various training metrics from ARS training."""
    data = load_training_data(logdir)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Plot average reward over iterations
    ax1 = fig.add_subplot(221)
    iterations = range(len(data['AverageReturn']))
    ax1.plot(iterations, data['AverageReturn'], 'b-', label='Average Return')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Average Return')
    ax1.set_title('Training Progress')
    ax1.grid(True)
    
    # 2. Plot max reward over iterations
    ax2 = fig.add_subplot(222)
    ax2.plot(iterations, data['MaxReturn'], 'r-', label='Max Return')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Max Return')
    ax2.set_title('Best Performance')
    ax2.grid(True)
    
    # 3. Plot standard deviation of returns
    ax3 = fig.add_subplot(223)
    ax3.plot(iterations, data['StdReturn'], 'g-', label='Std Return')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Return Variability')
    ax3.grid(True)
    
    # 4. Plot total timesteps
    ax4 = fig.add_subplot(224)
    ax4.plot(iterations, data['TimestepsSoFar'], 'k-', label='Total Timesteps')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Timesteps')
    ax4.set_title('Training Duration')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    else:
        plt.show()

def plot_policy_weights(logdir, save_dir=None):
    """Plot the evolution of policy weights over time."""
    data = load_training_data(logdir)
    
    if 'PolicyWeights' in data:
        weights = np.array(data['PolicyWeights'])
        
        plt.figure(figsize=(12, 6))
        plt.imshow(weights, aspect='auto', cmap='viridis')
        plt.colorbar(label='Weight Value')
        plt.xlabel('Weight Index')
        plt.ylabel('Iteration')
        plt.title('Policy Weights Evolution')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'policy_weights.png'))
        else:
            plt.show()

def main():
    # Get the log directory from environment variable or use default
    logdir = os.getenv('ARS_LOG_DIR', 'logs')
    save_dir = os.getenv('PLOT_SAVE_DIR', 'plots')
    
    try:
        plot_training_metrics(logdir, save_dir)
        plot_policy_weights(logdir, save_dir)
        print(f"Plots have been saved to {save_dir}")
    except Exception as e:
        print(f"Error plotting metrics: {str(e)}")

if __name__ == "__main__":
    main() 