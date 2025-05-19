"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse. 
Example usage:
python3 pupper_ars_run_policy.py --expert_policy_file=data/lin_policy_plus_best_10.npz --json_file=data/params.json

"""
import numpy as np
import gym
import time
import pybullet_envs
try:
  import tds_environments
except:
  pass
import json
from arspb.policies import *
import time
import arspb.trained_policies as tp
import os
import pickle
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
from pybullet import COV_ENABLE_GUI
import puppersim.data as pd
import argparse

def create_pupper_env():
    CONFIG_DIR = puppersim.getPupperSimPath()
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
    gin.parse_config_file(_CONFIG_FILE)
    env = env_loader.load()
    return env

def load_policy(policy_file, params):
    """Load policy from file."""
    print('loading and building expert policy')
    
    # Load the policy weights
    data = np.load(policy_file, allow_pickle=True)
    lst = data.files
    weights = data[lst[0]][0]
    mu = data[lst[0]][1]
    std = data[lst[0]][2]
    
    # Create environment to get dimensions
    env = create_pupper_env()
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    ac_lb = env.action_space.low
    ac_ub = env.action_space.high
    
    # Set up policy parameters
    policy_params = {
        'type': params['policy_type'],
        'ob_filter': params['filter'],
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'action_lower_bound': ac_lb,
        'action_upper_bound': ac_ub,
        'weights': weights,
        'observation_filter_mean': mu,
        'observation_filter_std': std
    }
    
    # Add network size for neural network policy
    if params['policy_type'] == 'nn':
        policy_sizes_list = [int(item) for item in params['policy_network_size_list'].split(',')]
        policy_params['policy_network_size'] = policy_sizes_list
        policy = FullyConnectedNeuralNetworkPolicy(policy_params, update_filter=False)
    else:
        policy = LinearPolicy2(policy_params, update_filter=False)
    
    return policy

def run_policy(policy, env, render=True, num_steps=1000):
    """Run policy in environment."""
    obs = env.reset()
    total_reward = 0
    
    for _ in range(num_steps):
        if render:
            env.render()
        
        action = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return total_reward

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str, required=True,
                      help='Path to the policy file (.npz for ARS, .pt for PPO)')
    parser.add_argument('--json_file', type=str, required=True,
                      help='Path to the parameters JSON file')
    parser.add_argument('--render', action='store_true',
                      help='Whether to render the environment')
    parser.add_argument('--num_steps', type=int, default=1000,
                      help='Number of steps to run the policy')
    args = parser.parse_args(argv)
    
    # Load parameters
    with open(args.json_file, 'r') as f:
        params = json.load(f)
    
    # Add env_name if not present
    if 'env_name' not in params:
        params['env_name'] = 'PupperEnv-v0'
    
    print("params=", params)
    
    # Create environment
    env = create_pupper_env()
    
    # Load and run policy
    policy = load_policy(args.expert_policy_file, params)
    total_reward = run_policy(policy, env, args.render, args.num_steps)
    
    print(f'Total reward: {total_reward}')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
