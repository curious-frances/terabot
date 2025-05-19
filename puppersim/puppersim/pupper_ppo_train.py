'''
Proximal Policy Optimization (PPO) implementation for quadruped robot training.
'''

import parser
import time
import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import ray
from packaging import version

import arspb.logz as logz
import arspb.utils as utils
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

def create_pupper_env():
    CONFIG_DIR = puppersim.getPupperSimPath()
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
    gin.parse_config_file(_CONFIG_FILE)
    env = env_loader.load()
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
    
    def evaluate_actions(self, obs, actions):
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()
        return value, log_prob, entropy

@ray.remote
class PPOWorker:
    def __init__(self, env_seed, policy_params=None):
        self.env = create_pupper_env()
        self.env.seed(env_seed)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy
        self.policy = ActorCritic(self.obs_dim, self.action_dim).to(self.device)
        if policy_params is not None:
            self.policy.load_state_dict(policy_params)
            
    def collect_rollout(self, max_steps=1000):
        obs = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        for _ in range(max_steps):
            with torch.no_grad():
                action = self.policy.get_action(obs)
                value, log_prob, _ = self.policy.evaluate_actions(obs.unsqueeze(0), action.unsqueeze(0))
            
            next_obs, reward, done, _ = self.env.step(action.cpu().numpy())
            next_obs = torch.FloatTensor(next_obs).to(self.device)
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            
            obs = next_obs
            if done:
                break
                
        return {
            'observations': torch.stack(observations),
            'actions': torch.stack(actions),
            'rewards': torch.tensor(rewards),
            'values': torch.stack(values).squeeze(),
            'log_probs': torch.stack(log_probs).squeeze()
        }
    
    def update_policy(self, policy_params):
        self.policy.load_state_dict(policy_params)
        return True

class PPOTrainer:
    def __init__(self, 
                 num_workers=32,
                 num_epochs=10,
                 batch_size=64,
                 clip_ratio=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 learning_rate=3e-4,
                 gamma=0.99,
                 lam=0.95,
                 logdir=None,
                 params=None):
        
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.logdir = logdir
        self.params = params
        
        # Initialize environment and policy
        env = create_pupper_env()
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Initialize workers
        self.workers = [PPOWorker.remote(i) for i in range(num_workers)]
        
        # Setup logging
        if logdir:
            logz.configure_output_dir(logdir)
            logz.save_params(params)
            
        # Track best policy
        self.best_mean_reward = float('-inf')
            
    def compute_advantages(self, rewards, values, next_value):
        advantages = []
        gae = 0
        
        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.gamma * next_value - v
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            next_value = v
            
        advantages = torch.tensor(advantages)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, rollouts):
        obs = rollouts['observations']
        actions = rollouts['actions']
        old_values = rollouts['values']
        old_log_probs = rollouts['log_probs']
        rewards = rollouts['rewards']
        
        # Compute advantages
        with torch.no_grad():
            next_value = self.policy.critic(obs[-1].unsqueeze(0)).squeeze()
        advantages, returns = self.compute_advantages(rewards, old_values, next_value)
        
        # Update policy for multiple epochs
        for _ in range(self.num_epochs):
            # Create mini-batches
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                idx = indices[start:start + self.batch_size]
                
                # Get current policy predictions
                values, log_probs, entropy = self.policy.evaluate_actions(
                    obs[idx], actions[idx])
                
                # Compute PPO loss
                ratio = torch.exp(log_probs - old_log_probs[idx])
                clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages[idx]
                policy_loss = -torch.min(ratio * advantages[idx], clip_adv).mean()
                
                # Compute value loss
                value_loss = 0.5 * (values - returns[idx]).pow(2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def train(self, num_iterations):
        for i in range(num_iterations):
            # Collect rollouts from all workers
            rollout_ids = [worker.collect_rollout.remote() for worker in self.workers]
            rollouts = ray.get(rollout_ids)
            
            # Update policy
            metrics = self.update_policy(rollouts[0])  # Using first worker's data for now
            
            # Log metrics
            if (i + 1) % 10 == 0:
                # Evaluate current policy
                eval_rollouts = ray.get([worker.collect_rollout.remote() for worker in self.workers[:5]])  # Use 5 workers for evaluation
                mean_reward = np.mean([np.sum(r['rewards']) for r in eval_rollouts])
                
                # Save latest policy
                if self.logdir:
                    torch.save(self.policy.state_dict(), os.path.join(self.logdir, 'policy_latest.pt'))
                    # Also save as npz for consistency with ARS
                    policy_data = {
                        'weights': self.policy.state_dict(),
                        'mu': torch.zeros(self.obs_dim),  # Placeholder for observation filter mean
                        'std': torch.ones(self.obs_dim)   # Placeholder for observation filter std
                    }
                    torch.save(policy_data, os.path.join(self.logdir, 'policy_plus_latest.npz'))
                
                # Save best policy if better
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.logdir:
                        torch.save(self.policy.state_dict(), os.path.join(self.logdir, 'policy_best.pt'))
                        # Also save as npz for consistency with ARS
                        policy_data = {
                            'weights': self.policy.state_dict(),
                            'mu': torch.zeros(self.obs_dim),  # Placeholder for observation filter mean
                            'std': torch.ones(self.obs_dim)   # Placeholder for observation filter std
                        }
                        torch.save(policy_data, os.path.join(self.logdir, f'policy_plus_best_{i+1}.npz'))
                
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("PolicyLoss", metrics['policy_loss'])
                logz.log_tabular("ValueLoss", metrics['value_loss'])
                logz.log_tabular("Entropy", metrics['entropy'])
                logz.log_tabular("MeanReward", mean_reward)
                logz.log_tabular("BestMeanReward", self.best_mean_reward)
                logz.dump_tabular()
                
            # Sync policy to workers
            policy_params = ray.put(self.policy.state_dict())
            ray.get([worker.update_policy.remote(policy_params) for worker in self.workers])

def run_ppo(params):
    dir_path = params['dir_path']
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Create timestamped directory for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join(dir_path, f"ppo_{timestamp}")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    try:
        import pybullet_envs
    except:
        pass
    try:
        import tds_environments
    except:
        pass
        
    PPO = PPOTrainer(
        num_workers=params['n_workers'],
        num_epochs=params['n_epochs'],
        batch_size=params['batch_size'],
        clip_ratio=params['clip_ratio'],
        value_coef=params['value_coef'],
        entropy_coef=params['entropy_coef'],
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        lam=params['lam'],
        logdir=logdir,
        params=params
    )
    
    PPO.train(params['n_iter'])
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_workers', '-e', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--dir_path', type=str, default='data')
    args = parser.parse_args()
    params = vars(args)
    
    ray.init()
    assert ray.is_initialized()
    try:
        run_ppo(params)
    finally:
        ray.shutdown() 