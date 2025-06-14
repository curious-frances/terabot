import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import argparse
import matplotlib as mpl
import json

class LogParser:
    """Base class for log parsing."""
    def __init__(self):
        self.metrics = {}
    
    def parse_line(self, line):
        """Parse a single line of log data."""
        raise NotImplementedError
    
    def get_metrics(self):
        """Return the parsed metrics."""
        return self.metrics

class LinearPolicyLogParser(LogParser):
    """Parser for linear policy training logs."""
    def __init__(self):
        super().__init__()
        self.metrics = {
            'iterations': [],
            'PolicyLoss': [],
            'ValueLoss': [],
            'Entropy': []
        }
    
    def parse_line(self, line):
        """Parse a linear policy log line."""
        parts = line.strip().split('\t')
        try:
            if len(parts) >= 4:  # We expect at least 4 columns
                self.metrics['iterations'].append(int(parts[0]))
                self.metrics['PolicyLoss'].append(float(parts[1]))
                self.metrics['ValueLoss'].append(float(parts[2]))
                self.metrics['Entropy'].append(float(parts[3]))
                return True
        except (ValueError, IndexError):
            pass
        return False

class PPOLogParser(LogParser):
    """Parser for PPO training logs."""
    def __init__(self):
        super().__init__()
        self.metrics = {
            'iterations': [],
            'PolicyLoss': [],
            'ValueLoss': [],
            'Entropy': [],
            'AverageReward': [],
            'StdRewards': [],
            # New metrics
            'ClipFraction': [],
            'GradientNorm': [],
            'HeightError': [],
            'PitchError': [],
            'RollError': [],
            'JointVelocity': [],
            'EnergyConsumption': [],
            'SuccessRate': [],
            'EpisodeLength': [],
            'UpdateTime': [],
            'RolloutTime': [],
            'LearningRate': []
        }
    
    def parse_line(self, line):
        """Parse a PPO log line."""
        parts = line.strip().split('\t')
        try:
            # Basic metrics (maintain backward compatibility)
            self.metrics['iterations'].append(int(parts[0]))
            self.metrics['PolicyLoss'].append(float(parts[1]))
            self.metrics['ValueLoss'].append(float(parts[2]))
            self.metrics['Entropy'].append(float(parts[3]))
            
            # Extended metrics (if available)
            if len(parts) > 4:
                self.metrics['AverageReward'].append(float(parts[4]))
                self.metrics['StdRewards'].append(float(parts[5]))
                
                # New metrics (if available in log)
                if len(parts) > 6:
                    self.metrics['ClipFraction'].append(float(parts[6]))
                    self.metrics['GradientNorm'].append(float(parts[7]))
                    self.metrics['HeightError'].append(float(parts[8]))
                    self.metrics['PitchError'].append(float(parts[9]))
                    self.metrics['RollError'].append(float(parts[10]))
                    self.metrics['JointVelocity'].append(float(parts[11]))
                    self.metrics['EnergyConsumption'].append(float(parts[12]))
                    self.metrics['SuccessRate'].append(float(parts[13]))
                    self.metrics['EpisodeLength'].append(float(parts[14]))
                    self.metrics['UpdateTime'].append(float(parts[15]))
                    self.metrics['RolloutTime'].append(float(parts[16]))
                    self.metrics['LearningRate'].append(float(parts[17]))
            return True
        except (ValueError, IndexError):
            return False

class ARSLogParser(LogParser):
    """Parser for ARS training logs."""
    def __init__(self):
        super().__init__()
        self.metrics = {
            'iterations': [],
            'AverageReward': [],
            'StdRewards': [],
            'MaxReward': [],
            'MinReward': [],
            'Timesteps': []
        }
    
    def parse_line(self, line):
        """Parse an ARS log line."""
        parts = line.strip().split('\t')
        try:
            self.metrics['iterations'].append(int(parts[0]))
            self.metrics['AverageReward'].append(float(parts[1]))
            self.metrics['StdRewards'].append(float(parts[2]))
            self.metrics['MaxReward'].append(float(parts[3]))
            self.metrics['MinReward'].append(float(parts[4]))
            self.metrics['Timesteps'].append(int(parts[5]))
            return True
        except (ValueError, IndexError):
            return False

def load_training_data(logdir):
    """Load training data from log directory."""
    log_file = os.path.join(logdir, "log.txt")
    if not os.path.exists(log_file):
        raise ValueError(f"No log.txt file found in {log_file}")
    
    print(f"Loading data from: {log_file}")
    
    # Try different parsers
    parsers = [LinearPolicyLogParser(), PPOLogParser(), ARSLogParser()]
    successful_parser = None
    
    with open(log_file, 'r') as f:
        # Skip header line
        next(f)
        # Try each parser on the first few lines
        for parser in parsers:
            f.seek(0)
            next(f)  # Skip header
            success_count = 0
            for _ in range(5):  # Try first 5 lines
                line = next(f, None)
                if line and parser.parse_line(line):
                    success_count += 1
            if success_count >= 3:  # If at least 3 lines parsed successfully
                successful_parser = parser
                break
    
    if not successful_parser:
        raise ValueError("Could not determine log format")
    
    # Parse all lines with the successful parser
    with open(log_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            if not successful_parser.parse_line(line):
                print(f"Warning: Skipping malformed line: {line.strip()}")
    
    return successful_parser.get_metrics()

def setup_publication_style():
    """Configure matplotlib for publication-quality plots."""
    # Use a basic style that's guaranteed to exist
    plt.style.use('default')
    
    # Configure the style manually
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
    
    # Add grid
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['grid.alpha'] = 0.3

def plot_training_metrics(logdir, save_dir=None, run_name=None):
    """Plot various training metrics from training."""
    data = load_training_data(logdir)
    setup_publication_style()
    
    # Define color scheme for better visibility
    colors = {
        'Average Reward': '#1f77b4',  # Blue
        'Policy Loss': '#d62728',     # Red
        'Value Loss': '#2ca02c',      # Green
        'Entropy': '#7f7f7f',         # Gray
        'Max Reward': '#ff7f0e',      # Orange
        'Min Reward': '#9467bd',      # Purple
        'Timesteps': '#8c564b',       # Brown
        'Clip Fraction': '#e377c2',   # Pink
        'Gradient Norm': '#17becf',   # Cyan
        'Height Error': '#bcbd22',    # Yellow
        'Pitch Error': '#ff9896',     # Light Red
        'Roll Error': '#98df8a',      # Light Green
        'Joint Velocity': '#c5b0d5',  # Light Purple
        'Energy Consumption': '#ffbb78', # Light Orange
        'Success Rate': '#aec7e8',    # Light Blue
        'Episode Length': '#ff7f0e',  # Orange
        'Update Time': '#2ca02c',     # Green
        'Rollout Time': '#d62728',    # Red
        'Learning Rate': '#9467bd'    # Purple
    }
    
    # Create plots for available metrics
    metrics = {}
    if 'AverageReward' in data and len(data['AverageReward']) > 0:
        metrics['Average Reward'] = ('AverageReward', colors['Average Reward'])
    if 'PolicyLoss' in data and len(data['PolicyLoss']) > 0:
        metrics['Policy Loss'] = ('PolicyLoss', colors['Policy Loss'])
    if 'ValueLoss' in data and len(data['ValueLoss']) > 0:
        metrics['Value Loss'] = ('ValueLoss', colors['Value Loss'])
    if 'Entropy' in data and len(data['Entropy']) > 0:
        metrics['Entropy'] = ('Entropy', colors['Entropy'])
    if 'MaxReward' in data and len(data['MaxReward']) > 0:
        metrics['Max Reward'] = ('MaxReward', colors['Max Reward'])
    if 'MinReward' in data and len(data['MinReward']) > 0:
        metrics['Min Reward'] = ('MinReward', colors['Min Reward'])
    if 'Timesteps' in data and len(data['Timesteps']) > 0:
        metrics['Timesteps'] = ('Timesteps', colors['Timesteps'])
    
    # Add new metrics if available
    if 'ClipFraction' in data and len(data['ClipFraction']) > 0:
        metrics['Clip Fraction'] = ('ClipFraction', colors['Clip Fraction'])
    if 'GradientNorm' in data and len(data['GradientNorm']) > 0:
        metrics['Gradient Norm'] = ('GradientNorm', colors['Gradient Norm'])
    if 'HeightError' in data and len(data['HeightError']) > 0:
        metrics['Height Error'] = ('HeightError', colors['Height Error'])
    if 'PitchError' in data and len(data['PitchError']) > 0:
        metrics['Pitch Error'] = ('PitchError', colors['Pitch Error'])
    if 'RollError' in data and len(data['RollError']) > 0:
        metrics['Roll Error'] = ('RollError', colors['Roll Error'])
    if 'JointVelocity' in data and len(data['JointVelocity']) > 0:
        metrics['Joint Velocity'] = ('JointVelocity', colors['Joint Velocity'])
    if 'EnergyConsumption' in data and len(data['EnergyConsumption']) > 0:
        metrics['Energy Consumption'] = ('EnergyConsumption', colors['Energy Consumption'])
    if 'SuccessRate' in data and len(data['SuccessRate']) > 0:
        metrics['Success Rate'] = ('SuccessRate', colors['Success Rate'])
    if 'EpisodeLength' in data and len(data['EpisodeLength']) > 0:
        metrics['Episode Length'] = ('EpisodeLength', colors['Episode Length'])
    if 'UpdateTime' in data and len(data['UpdateTime']) > 0:
        metrics['Update Time'] = ('UpdateTime', colors['Update Time'])
    if 'RolloutTime' in data and len(data['RolloutTime']) > 0:
        metrics['Rollout Time'] = ('RolloutTime', colors['Rollout Time'])
    if 'LearningRate' in data and len(data['LearningRate']) > 0:
        metrics['Learning Rate'] = ('LearningRate', colors['Learning Rate'])
    
    if not metrics:
        raise ValueError("No valid metrics found in the log file")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if run_name:
            save_dir = os.path.join(save_dir, run_name)
            os.makedirs(save_dir, exist_ok=True)
    
    # Create a figure with subplots for related metrics
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 2)
    
    # Group metrics by category
    metric_groups = {
        'Performance': ['Average Reward', 'Success Rate', 'Episode Length'],
        'Losses': ['Policy Loss', 'Value Loss', 'Entropy'],
        'Stability': ['Height Error', 'Pitch Error', 'Roll Error'],
        'Efficiency': ['Energy Consumption', 'Joint Velocity'],
        'Training': ['Clip Fraction', 'Gradient Norm', 'Learning Rate'],
        'Timing': ['Update Time', 'Rollout Time']
    }
    
    # Plot each group in a subplot
    for i, (group_name, group_metrics) in enumerate(metric_groups.items()):
        ax = fig.add_subplot(gs[i//2, i%2])
        for metric_name in group_metrics:
            if metric_name in metrics:
                metric_key, color = metrics[metric_name]
                ax.plot(data['iterations'], data[metric_key], color=color, label=metric_name)
                
                # Add confidence interval for reward
                if metric_key == 'AverageReward' and 'StdRewards' in data:
                    ax.fill_between(data['iterations'],
                                  np.array(data[metric_key]) - np.array(data['StdRewards']),
                                  np.array(data[metric_key]) + np.array(data['StdRewards']),
                                  color=color, alpha=0.2)
        
        ax.set_title(group_name, fontweight='bold', pad=20)
        ax.set_xlabel('Iteration', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_dir:
        # Save high-quality figure
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'),
                   dpi=300, bbox_inches='tight', pad_inches=0.1)
        # Also save as PDF for publications
        plt.savefig(os.path.join(save_dir, 'training_metrics.pdf'),
                   bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()
    
    # Also save individual plots for each metric
    for title, (metric, color) in metrics.items():
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Plot main line
        plt.plot(data['iterations'], data[metric], color=color, label=title, linewidth=2)
        
        # Add confidence interval for reward
        if metric == 'AverageReward' and 'StdRewards' in data and len(data['StdRewards']) > 0:
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