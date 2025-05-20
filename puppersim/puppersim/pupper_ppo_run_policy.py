"""
Code to load a PPO policy and generate rollout data.
Example usage:
python3 pupper_ppo_run_policy.py --expert_policy_file=data/ppo_20250519_172611/policy_plus_best_830.npz --json_file=data/params.json --render
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
import torch
import torch.nn as nn
from torch.distributions import Normal
import time
import os
import pickle
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
from pybullet import COV_ENABLE_GUI
import puppersim.data as pd
import argparse

def create_pupper_env(args):
    CONFIG_DIR = puppersim.getPupperSimPath()
    if args.run_on_robot:
        _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg_robot.gin")
    else:
        _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
    gin.parse_config_file(_CONFIG_FILE)
    gin.bind_parameter("SimulationParameters.enable_rendering", args.render)
    env = env_loader.load()
    env._pybullet_client.configureDebugVisualizer(COV_ENABLE_GUI, 0)
    return env

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Actor (Policy) Network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic (Value) Network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable standard deviation for action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, x):
        return self.actor(x), self.critic(x)
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            action_mean, _ = self.forward(obs)
            if deterministic:
                return action_mean
            std = torch.exp(self.log_std)
            dist = Normal(action_mean, std)
            action = dist.sample()
        return action

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str, required=True,
                      help='Path to the policy file (.npz or .pt)')
    parser.add_argument('--json_file', type=str, required=True,
                      help='Path to the parameters JSON file')
    parser.add_argument('--run_on_robot', action='store_true',
                      help='whether to run the policy on the robot instead of in simulation. Default is False.')
    parser.add_argument('--render', default=False, action='store_true',
                      help='whether to render the robot. Default is False.')
    parser.add_argument('--profile', default=False, action='store_true',
                      help='whether to print timing for parts of the code. Default is False.')
    parser.add_argument('--plot', default=False, action='store_true',
                      help='whether to plot action and observation histories after running the policy.')
    parser.add_argument("--log_to_file", default=False, action='store_true',
                      help="Whether to log data to the disk.")
    parser.add_argument("--realtime", default=False, help="Run at realtime.")
    args = parser.parse_args(argv)

    print('loading and building expert policy')
    with open(args.json_file) as f:
        params = json.load(f)
    
    # Add env_name if not present
    if 'env_name' not in params:
        params['env_name'] = 'PupperEnv-v0'
    
    print("params=", params)
    
    # Create environment
    env = create_pupper_env(args)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize policy
    policy = ActorCritic(obs_dim, action_dim).to(device)
    
    # Load policy weights
    if args.expert_policy_file.endswith('.npz'):
        data = torch.load(args.expert_policy_file)
        policy.load_state_dict(data['weights'])
    else:  # .pt file
        policy.load_state_dict(torch.load(args.expert_policy_file))
    
    returns = []
    observations = []
    actions = []

    log_dict = {
        't': [],
        'IMU': [],
        'MotorAngle': [],
        'action': []
    }

    try:
        obs = env.reset()
        obs = torch.FloatTensor(obs).to(device)
        done = False
        totalr = 0.
        steps = 0
        start_time_wall = time.time()
        env_start_time_wall = time.time()
        last_spammy_log = 0.0
        
        while not done or args.run_on_robot:
            if args.realtime or args.run_on_robot:  # always run at realtime with real robot
                # Sync to real time.
                wall_elapsed = time.time() - env_start_time_wall
                sim_elapsed = env.env_step_counter * env.env_time_step
                sleep_time = sim_elapsed - wall_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -1 and time.time() - last_spammy_log > 1.0:
                    print(f"Cannot keep up with realtime. {-sleep_time:.2f} sec behind, "
                          f"sim/wall ratio {(sim_elapsed/wall_elapsed):.2f}.")
                    last_spammy_log = time.time()

            if args.profile:
                print("loop dt:", time.time() - start_time_wall)
            start_time_wall = time.time()
            before_policy = time.time()
            
            # Get action from policy
            with torch.no_grad():
                action = policy.get_action(obs, deterministic=True)
            
            after_policy = time.time()

            if not args.run_on_robot:
                observations.append(obs.cpu().numpy())
                actions.append(action.cpu().numpy())

            next_obs, r, done, _ = env.step(action.cpu().numpy())
            next_obs = torch.FloatTensor(next_obs).to(device)
            
            if args.log_to_file:
                log_dict['t'].append(env.robot.GetTimeSinceReset())
                log_dict['MotorAngle'].append(next_obs.cpu().numpy()[0:12])
                log_dict['IMU'].append(next_obs.cpu().numpy()[12:16])
                log_dict['action'].append(action.cpu().numpy())

            totalr += r
            steps += 1
            obs = next_obs
            
            if args.profile:
                print('policy.act(obs): ', after_policy - before_policy)
                print('wallclock_control_code: ', time.time() - start_time_wall)
                
        returns.append(totalr)
    finally:
        if args.log_to_file:
            print("logging to file...")
            with open("env_ppo_log.txt", "wb") as f:
                pickle.dump(log_dict, f)

    print('returns: ', returns)
    print('mean return: ', np.mean(returns))
    print('std of return: ', np.std(returns))

    if args.plot and not args.run_on_robot:
        import matplotlib.pyplot as plt
        action_history = np.array(actions)
        observation_history = np.array(observations)
        plt.plot(action_history)
        plt.show()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:]) 