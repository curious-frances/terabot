import os
import time
import numpy as np
import torch
import json
import argparse
from reacher_ars_train import run_ars
from reacher_sac_train import train_sac
from reacher_ppo_train import train_ppo
import matplotlib.pyplot as plt

def run_comparison(params):
    # Create directories for results
    ars_dir = os.path.join(params['dir_path'], 'ars')
    sac_dir = os.path.join(params['dir_path'], 'sac')
    ppo_dir = os.path.join(params['dir_path'], 'ppo')
    os.makedirs(ars_dir, exist_ok=True)
    os.makedirs(sac_dir, exist_ok=True)
    os.makedirs(ppo_dir, exist_ok=True)
    
    # Run ARS
    print("Running ARS...")
    ars_params = {
        'env_name': 'Reacher-v1',
        'n_iter': params['n_iter'],
        'n_directions': 16,
        'deltas_used': 16,
        'step_size': 0.03,
        'delta_std': 0.03,
        'n_workers': 18,
        'rollout_length': 400,
        'shift': 0,
        'seed': params['seed'],
        'policy_type': 'nn',
        'dir_path': ars_dir,
        'filter': 'MeanStdFilter',
        'activation': 'tanh',
        'policy_network_size_list': '64,64'
    }
    run_ars(ars_params)
    
    # Run SAC
    print("Running SAC...")
    sac_params = {
        'n_iter': params['n_iter'],
        'dir_path': sac_dir,
        'seed': params['seed'],
        'start_time': time.time()
    }
    train_sac(sac_params)
    
    # Run PPO
    print("Running PPO...")
    ppo_params = {
        'n_iter': params['n_iter'],
        'dir_path': ppo_dir,
        'seed': params['seed'],
        'start_time': time.time()
    }
    train_ppo(ppo_params)
    
    # Plot comparison
    plot_comparison(ars_dir, sac_dir, ppo_dir, params['dir_path'])

def plot_individual_algorithm(data, algorithm_name, output_dir, window_size=10):
    plt.figure(figsize=(10, 6))
    
    # Plot raw data
    plt.plot(data, label='Raw', alpha=0.5)
    
    # Plot moving average
    ma = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(data)), ma, label='Moving Average', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title(f'{algorithm_name} Performance')
    plt.legend()
    plt.grid(True)
    
    # Add statistics
    stats_text = f'Mean: {np.mean(data):.2f}\nStd: {np.std(data):.2f}\nMax: {np.max(data):.2f}\nMin: {np.min(data):.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(output_dir, f'{algorithm_name.lower()}_performance.png'))
    plt.close()

def plot_comparison(ars_dir, sac_dir, ppo_dir, output_dir):
    # Load ARS data
    ars_data = []
    for filename in os.listdir(ars_dir):
        if filename.endswith('.log'):
            with open(os.path.join(ars_dir, filename), 'r') as f:
                for line in f:
                    if line.startswith('AverageReward'):
                        ars_data.append(float(line.split('\t')[1]))
    
    # Load SAC data
    sac_data = []
    for filename in os.listdir(sac_dir):
        if filename.endswith('.log'):
            with open(os.path.join(sac_dir, filename), 'r') as f:
                for line in f:
                    if line.startswith('AverageReward'):
                        sac_data.append(float(line.split('\t')[1]))
    
    # Load PPO data
    ppo_data = []
    for filename in os.listdir(ppo_dir):
        if filename.endswith('.log'):
            with open(os.path.join(ppo_dir, filename), 'r') as f:
                for line in f:
                    if line.startswith('AverageReward'):
                        ppo_data.append(float(line.split('\t')[1]))
    
    # Plot individual algorithms
    plot_individual_algorithm(ars_data, 'ARS', output_dir)
    plot_individual_algorithm(sac_data, 'SAC', output_dir)
    plot_individual_algorithm(ppo_data, 'PPO', output_dir)
    
    # Plot combined comparison
    plt.figure(figsize=(12, 8))
    plt.plot(ars_data, label='ARS', alpha=0.7)
    plt.plot(sac_data, label='SAC', alpha=0.7)
    plt.plot(ppo_data, label='PPO', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('ARS vs SAC vs PPO Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    # Add moving average
    window_size = 10
    plt.plot(np.convolve(ars_data, np.ones(window_size)/window_size, mode='valid'), 
             label='ARS (MA)', linestyle='--')
    plt.plot(np.convolve(sac_data, np.ones(window_size)/window_size, mode='valid'), 
             label='SAC (MA)', linestyle='--')
    plt.plot(np.convolve(ppo_data, np.ones(window_size)/window_size, mode='valid'), 
             label='PPO (MA)', linestyle='--')
    
    plt.savefig(os.path.join(output_dir, 'comparison.png'))
    plt.close()
    
    # Save numerical data
    comparison_data = {
        'ars_rewards': ars_data,
        'sac_rewards': sac_data,
        'ppo_rewards': ppo_data,
        'ars_mean': np.mean(ars_data),
        'ars_std': np.std(ars_data),
        'sac_mean': np.mean(sac_data),
        'sac_std': np.std(sac_data),
        'ppo_mean': np.mean(ppo_data),
        'ppo_std': np.std(ppo_data)
    }
    
    with open(os.path.join(output_dir, 'comparison_data.json'), 'w') as f:
        json.dump(comparison_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', '-n', type=int, default=50000)
    parser.add_argument('--dir_path', type=str, default='data/comparison')
    parser.add_argument('--seed', type=int, default=37)
    args = parser.parse_args()
    params = vars(args)
    
    run_comparison(params) 