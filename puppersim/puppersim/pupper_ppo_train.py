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
import json

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

def load_ars_policy(ars_policy_path, params_path):
    """Load ARS policy weights and parameters."""
    with open(params_path) as f:
        params = json.load(f)
    
    data = np.load(ars_policy_path, allow_pickle=True)
    lst = data.files
    weights = data[lst[0]][0]
    mu = data[lst[0]][1]
    std = data[lst[0]][2]
    
    return weights, mu, std, params

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, ars_weights=None, obs_filter=None):
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
        
        # Initialize with ARS weights if provided
        if ars_weights is not None:
            self.initialize_from_ars(ars_weights)
            
        # Observation filter
        self.obs_filter = obs_filter
        
    def initialize_from_ars(self, ars_weights):
        """Initialize policy network weights from ARS policy."""
        # Convert ARS weights to PyTorch format and initialize actor network
        with torch.no_grad():
            # Assuming ARS weights are in the correct format
            # You might need to adjust this based on your ARS policy structure
            self.actor[0].weight.data = torch.FloatTensor(ars_weights[:self.actor[0].weight.shape[0], :self.actor[0].weight.shape[1]])
            self.actor[0].bias.data = torch.FloatTensor(ars_weights[:self.actor[0].weight.shape[0], -1])
            
    def forward(self, x):
        if self.obs_filter is not None:
            x = (x - self.obs_filter['mu']) / (self.obs_filter['std'] + 1e-8)
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
    def __init__(self, env_seed, policy_params=None, obs_filter=None):
        self.env = create_pupper_env()
        self.env.seed(env_seed)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy
        self.policy = ActorCritic(self.obs_dim, self.action_dim, obs_filter=obs_filter).to(self.device)
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
                 params=None,
                 ars_policy_path=None,
                 ars_params_path=None):
        
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
        
        # Load ARS policy if provided
        ars_weights = None
        obs_filter = None
        if ars_policy_path and ars_params_path:
            ars_weights, mu, std, _ = load_ars_policy(ars_policy_path, ars_params_path)
            obs_filter = {'mu': torch.FloatTensor(mu), 'std': torch.FloatTensor(std)}
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim, ars_weights=ars_weights, obs_filter=obs_filter).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Initialize workers
        self.workers = [PPOWorker.remote(i, obs_filter=obs_filter) for i in range(num_workers)]
        
        # Setup logging
        if logdir:
            logz.configure_output_dir(logdir)
            logz.save_params(params)
            
        # Track best policy
        self.best_mean_reward = float('-inf')
        
        # Initialize metrics tracking
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'mean_reward': [],
            'std_reward': [],
            'max_reward': [],
            'min_reward': [],
            'learning_rate': [],
            'timesteps': []
        }
            
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
        start_time = time.time()
        total_timesteps = 0
        
        for i in range(num_iterations):
            # Collect rollouts from all workers
            rollout_ids = [worker.collect_rollout.remote() for worker in self.workers]
            rollouts = ray.get(rollout_ids)
            
            # Update policy
            metrics = self.update_policy(rollouts[0])  # Using first worker's data for now
            
            # Evaluate current policy
            eval_rollouts = ray.get([worker.collect_rollout.remote() for worker in self.workers[:5]])
            eval_rewards = [rollout['rewards'].sum().item() for rollout in eval_rollouts]
            
            # Update metrics
            self.metrics['policy_loss'].append(metrics['policy_loss'])
            self.metrics['value_loss'].append(metrics['value_loss'])
            self.metrics['entropy'].append(metrics['entropy'])
            self.metrics['mean_reward'].append(np.mean(eval_rewards))
            self.metrics['std_reward'].append(np.std(eval_rewards))
            self.metrics['max_reward'].append(np.max(eval_rewards))
            self.metrics['min_reward'].append(np.min(eval_rewards))
            self.metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Update timesteps
            total_timesteps += len(rollouts[0]['rewards'])
            self.metrics['timesteps'].append(total_timesteps)
            
            # Log metrics
            if (i + 1) % 10 == 0:
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("Time", time.time() - start_time)
                logz.log_tabular("Timesteps", total_timesteps)
                logz.log_tabular("MeanReward", np.mean(eval_rewards))
                logz.log_tabular("StdReward", np.std(eval_rewards))
                logz.log_tabular("MaxReward", np.max(eval_rewards))
                logz.log_tabular("MinReward", np.min(eval_rewards))
                logz.log_tabular("PolicyLoss", metrics['policy_loss'])
                logz.log_tabular("ValueLoss", metrics['value_loss'])
                logz.log_tabular("Entropy", metrics['entropy'])
                logz.dump_tabular()
                
                # Save best policy
                if np.mean(eval_rewards) > self.best_mean_reward:
                    self.best_mean_reward = np.mean(eval_rewards)
                    if self.logdir:
                        torch.save(self.policy.state_dict(), 
                                 os.path.join(self.logdir, 'best_policy.pt'))
                        
                # Save metrics for plotting
                np.save(os.path.join(self.logdir, 'metrics.npy'), self.metrics)

def run_ppo(params):
    """Run PPO training with the given parameters."""
    # Create log directory
    logdir = os.path.join('data', 'ppo_' + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    
    # Initialize trainer
    trainer = PPOTrainer(
        num_workers=params.get('num_workers', 32),
        num_epochs=params.get('num_epochs', 10),
        batch_size=params.get('batch_size', 64),
        clip_ratio=params.get('clip_ratio', 0.2),
        value_coef=params.get('value_coef', 0.5),
        entropy_coef=params.get('entropy_coef', 0.01),
        learning_rate=params.get('learning_rate', 3e-4),
        gamma=params.get('gamma', 0.99),
        lam=params.get('lam', 0.95),
        logdir=logdir,
        params=params,
        ars_policy_path=params.get('ars_policy_path'),
        ars_params_path=params.get('ars_params_path')
    )
    
    # Train
    trainer.train(params.get('num_iterations', 1000))
    
    return logdir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ars_policy_path', type=str, default=None,
                      help='Path to ARS policy weights file')
    parser.add_argument('--ars_params_path', type=str, default=None,
                      help='Path to ARS parameters file')
    parser.add_argument('--num_workers', type=int, default=32,
                      help='Number of parallel workers')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='Number of epochs per update')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for updates')
    parser.add_argument('--num_iterations', type=int, default=1000,
                      help='Number of training iterations')
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init()
    
    # Set up parameters
    params = vars(args)
    
    # Run training
    logdir = run_ppo(params)
    print(f"Training complete. Logs saved to {logdir}") 