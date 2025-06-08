import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_training_curves(metrics_path, save_dir=None):
    """Plot comprehensive training curves from metrics."""
    # Load metrics
    if metrics_path.endswith('.npy'):
        metrics = np.load(metrics_path, allow_pickle=True).item()
    else:
        metrics = pd.read_csv(metrics_path)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 2)
    
    # 1. Performance Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics['timesteps'], metrics['mean_reward'], label='Mean Reward')
    ax1.fill_between(metrics['timesteps'],
                     metrics['mean_reward'] - metrics['std_reward'],
                     metrics['mean_reward'] + metrics['std_reward'],
                     alpha=0.2)
    ax1.set_title('Reward Progress')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Reward')
    ax1.legend()
    
    # 2. Success Rate and Episode Length
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(metrics['timesteps'], metrics['success_rate'], label='Success Rate')
    ax2.plot(metrics['timesteps'], metrics['episode_length'], label='Episode Length')
    ax2.set_title('Success Rate and Episode Length')
    ax2.set_xlabel('Timesteps')
    ax2.legend()
    
    # 3. Policy Losses
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(metrics['timesteps'], metrics['policy_loss'], label='Policy Loss')
    ax3.plot(metrics['timesteps'], metrics['value_loss'], label='Value Loss')
    ax3.set_title('Policy and Value Losses')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Loss')
    ax3.legend()
    
    # 4. Entropy and Clip Fraction
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(metrics['timesteps'], metrics['entropy'], label='Entropy')
    ax4.plot(metrics['timesteps'], metrics['clip_fraction'], label='Clip Fraction')
    ax4.set_title('Entropy and Clip Fraction')
    ax4.set_xlabel('Timesteps')
    ax4.legend()
    
    # 5. Stability Metrics
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(metrics['timesteps'], metrics['height_error'], label='Height Error')
    ax5.plot(metrics['timesteps'], metrics['pitch_error'], label='Pitch Error')
    ax5.plot(metrics['timesteps'], metrics['roll_error'], label='Roll Error')
    ax5.set_title('Stability Metrics')
    ax5.set_xlabel('Timesteps')
    ax5.set_ylabel('Error')
    ax5.legend()
    
    # 6. Energy and Smoothness
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(metrics['timesteps'], metrics['energy_consumption'], label='Energy')
    ax6.plot(metrics['timesteps'], metrics['joint_velocity'], label='Joint Velocity')
    ax6.set_title('Energy and Smoothness')
    ax6.set_xlabel('Timesteps')
    ax6.legend()
    
    # 7. Learning Rate and Gradient Norm
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(metrics['timesteps'], metrics['learning_rate'], label='Learning Rate')
    ax7.plot(metrics['timesteps'], metrics['gradient_norm'], label='Gradient Norm')
    ax7.set_title('Learning Rate and Gradient Norm')
    ax7.set_xlabel('Timesteps')
    ax7.legend()
    
    # 8. Timing Metrics
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(metrics['timesteps'], metrics['update_time'], label='Update Time')
    ax8.plot(metrics['timesteps'], metrics['rollout_time'], label='Rollout Time')
    ax8.set_title('Timing Metrics')
    ax8.set_xlabel('Timesteps')
    ax8.set_ylabel('Time (s)')
    ax8.legend()
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_action_distribution(metrics_path, save_dir=None):
    """Plot action distribution statistics."""
    metrics = pd.read_csv(metrics_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Action mean and std
    ax1.plot(metrics['timesteps'], metrics['action_mean'], label='Action Mean')
    ax1.fill_between(metrics['timesteps'],
                     metrics['action_mean'] - metrics['action_std'],
                     metrics['action_mean'] + metrics['action_std'],
                     alpha=0.2)
    ax1.set_title('Action Distribution')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Action Value')
    ax1.legend()
    
    # Value function statistics
    ax2.plot(metrics['timesteps'], metrics['value_mean'], label='Value Mean')
    ax2.fill_between(metrics['timesteps'],
                     metrics['value_mean'] - metrics['value_std'],
                     metrics['value_mean'] + metrics['value_std'],
                     alpha=0.2)
    ax2.set_title('Value Function Statistics')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Value')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'action_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_correlation_matrix(metrics_path, save_dir=None):
    """Plot correlation matrix of key metrics."""
    metrics = pd.read_csv(metrics_path)
    
    # Select key metrics for correlation
    key_metrics = [
        'mean_reward', 'success_rate', 'episode_length',
        'policy_loss', 'value_loss', 'entropy',
        'height_error', 'pitch_error', 'roll_error',
        'energy_consumption', 'joint_velocity'
    ]
    
    corr_matrix = metrics[key_metrics].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Metric Correlations')
    
    if save_dir:
        save_path = Path(save_dir) / 'correlation_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_path', type=str, required=True,
                      help='Path to metrics file (CSV or NPY)')
    parser.add_argument('--save_dir', type=str, default=None,
                      help='Directory to save plots')
    args = parser.parse_args()
    
    plot_training_curves(args.metrics_path, args.save_dir)
    plot_action_distribution(args.metrics_path, args.save_dir)
    plot_correlation_matrix(args.metrics_path, args.save_dir) 