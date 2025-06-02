import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import argparse
import matplotlib as mpl

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

def setup_publication_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-whitegrid')
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['figure.titlesize'] = 18
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.alpha'] = 0.3

def plot_training_metrics(logdir, save_dir=None, run_name=None):
    """Plot various training metrics from training."""
    data = load_training_data(logdir)
    setup_publication_style()
    
    # Define color scheme for better visibility
    colors = {
        'Average Reward': '#1f77b4',  # Blue
        'Max Reward': '#d62728',      # Red
        'Min Reward': '#2ca02c',      # Green
        'Standard Deviation': '#7f7f7f',  # Gray
        'Timesteps': '#9467bd',       # Purple
        'Training Time': '#17becf',   # Cyan
        'Update Norm': '#bcbd22',     # Yellow
        'Learning Rate': '#8c564b',   # Brown
        'Delta Std': '#e377c2',       # Pink
        'Deltas Used': '#ff7f0e',     # Orange
        'Success Rate': '#7f7f7f',    # Gray
        'Improvement': '#17becf'      # Cyan
    }
    
    # Create separate plots for each metric
    metrics = {
        'Average Reward': ('AverageReward', colors['Average Reward']),
        'Max Reward': ('MaxRewardRollout', colors['Max Reward']),
        'Min Reward': ('MinRewardRollout', colors['Min Reward']),
        'Standard Deviation': ('StdRewards', colors['Standard Deviation']),
        'Timesteps': ('timesteps', colors['Timesteps']),
        'Training Time': ('time', colors['Training Time']),
        'Update Norm': ('UpdateNorm', colors['Update Norm']),
        'Learning Rate': ('LearningRate', colors['Learning Rate']),
        'Delta Std': ('DeltaStd', colors['Delta Std']),
        'Deltas Used': ('DeltasUsed', colors['Deltas Used']),
        'Success Rate': ('SuccessRate', colors['Success Rate']),
        'Improvement': ('Improvement', colors['Improvement'])
    }
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if run_name:
            save_dir = os.path.join(save_dir, run_name)
            os.makedirs(save_dir, exist_ok=True)
    
    for title, (metric, color) in metrics.items():
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Plot main line
        plt.plot(data['iterations'], data[metric], color=color, label=title, linewidth=2)
        
        # Add confidence interval for reward-related metrics
        if 'Reward' in title:
            plt.fill_between(data['iterations'],
                           np.array(data[metric]) - np.array(data['StdRewards']),
                           np.array(data[metric]) + np.array(data['StdRewards']),
                           color=color, alpha=0.2)
        
        plt.xlabel('Iteration', fontweight='bold')
        plt.ylabel(title, fontweight='bold')
        plt.title(f'{title} Over Training', fontweight='bold', pad=20)
        
        # Customize grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend with better positioning
        plt.legend(loc='best', frameon=True, framealpha=0.9)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_dir:
            # Save high-quality figure
            plt.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}.png'),
                       dpi=300, bbox_inches='tight', pad_inches=0.1)
            # Also save as PDF for publications
            plt.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}.pdf'),
                       bbox_inches='tight', pad_inches=0.1)
        else:
            plt.show()
        plt.close()

def list_available_runs(data_dir):
    """List all available training runs in the data directory."""
    runs = []
    for run_dir in glob.glob(os.path.join(data_dir, "*_*")):
        if os.path.isdir(run_dir) and os.path.exists(os.path.join(run_dir, "log.txt")):
            run_name = os.path.basename(run_dir)
            runs.append(run_name)
    return sorted(runs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Base directory containing training runs')
    parser.add_argument('--run_name', type=str, default=None,
                      help='Specific run to plot (e.g., ars_20250518_202225). If not specified, will list available runs.')
    parser.add_argument('--save_dir', type=str, default='plots',
                      help='Directory to save plots')
    args = parser.parse_args()
    
    if args.run_name is None:
        # List available runs
        runs = list_available_runs(args.data_dir)
        if not runs:
            print(f"No training runs found in {args.data_dir}")
            return
        
        print("\nAvailable runs:")
        for i, run in enumerate(runs):
            print(f"{i+1}. {run}")
        
        # Ask user to select a run
        while True:
            try:
                choice = int(input("\nEnter the number of the run to plot (or 0 to exit): "))
                if choice == 0:
                    return
                if 1 <= choice <= len(runs):
                    args.run_name = runs[choice-1]
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    # Plot the selected run
    logdir = os.path.join(args.data_dir, args.run_name)
    try:
        plot_training_metrics(logdir, args.save_dir, args.run_name)
        print(f"Plots have been saved to {os.path.join(args.save_dir, args.run_name)}")
    except Exception as e:
        print(f"Error plotting metrics: {str(e)}")
        print("\nAvailable files in log directory:")
        for f in glob.glob(os.path.join(logdir, "*")):
            print(f"  - {f}")

if __name__ == "__main__":
    main() 