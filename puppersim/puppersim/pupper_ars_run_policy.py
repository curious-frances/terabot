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
    if params['policy_type'] == 'linear':
        policy = LinearPolicy2(params)
    else:
        policy = FullyConnectedNeuralNetworkPolicy(params)
    
    policy.load_weights(policy_file)
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
