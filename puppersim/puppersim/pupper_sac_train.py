'''
Soft Actor-Critic (SAC) implementation for quadruped robot training.
'''

import argparse
import time
import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.next_obs = np.zeros((max_size, obs_dim))
        self.dones = np.zeros((max_size, 1))
        
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.obs[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_obs[ind]),
            torch.FloatTensor(self.dones[ind])
        )

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128], obs_filter=None):
        super(GaussianPolicy, self).__init__()
        
        # Policy Network
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Linear(prev_dim, action_dim)
        
        # Observation filter
        self.obs_filter = obs_filter
        
    def forward(self, obs):
        if self.obs_filter is not None:
            obs = (obs - self.obs_filter['mu']) / (self.obs_filter['std'] + 1e-8)
        x = self.net(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        if deterministic:
            return mean
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Enforce action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128], obs_filter=None):
        super(QNetwork, self).__init__()
        
        # Q Network
        layers = []
        prev_dim = obs_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        self.net = nn.Sequential(*layers)
        self.q = nn.Linear(prev_dim, 1)
        
        # Observation filter
        self.obs_filter = obs_filter
        
    def forward(self, obs, action):
        if self.obs_filter is not None:
            obs = (obs - self.obs_filter['mu']) / (self.obs_filter['std'] + 1e-8)
        x = torch.cat([obs, action], dim=1)
        return self.q(self.net(x))

@ray.remote
class SACWorker:
    def __init__(self, env_seed, policy_params=None, obs_filter=None):
        self.env = create_pupper_env()
        self.env.seed(env_seed)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy
        self.policy = GaussianPolicy(self.obs_dim, self.action_dim, obs_filter=obs_filter).to(self.device)
        if policy_params is not None:
            self.policy.load_state_dict(policy_params)
            
    def collect_rollout(self, max_steps=1000):
        obs = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        
        for _ in range(max_steps):
            with torch.no_grad():
                action, _ = self.policy.sample(obs.unsqueeze(0))
                action = action.squeeze(0)
            
            next_obs, reward, done, _ = self.env.step(action.cpu().numpy())
            next_obs = torch.FloatTensor(next_obs).to(self.device)
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            dones.append(done)
            
            obs = next_obs
            if done:
                break
                
        return {
            'observations': torch.stack(observations),
            'actions': torch.stack(actions),
            'rewards': torch.tensor(rewards),
            'next_observations': torch.stack(next_observations),
            'dones': torch.tensor(dones)
        }
    
    def update_policy(self, policy_params):
        self.policy.load_state_dict(policy_params)
        return True

class SACTrainer:
    def __init__(self, 
                 num_workers=64,
                 batch_size=256,
                 buffer_size=int(1e6),
                 learning_rate=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 auto_entropy=True,
                 target_entropy=None,
                 logdir=None,
                 params=None,
                 ars_policy_path=None,
                 ars_params_path=None):
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        self.logdir = logdir
        self.params = params
        
        # Initialize environment and policy
        env = create_pupper_env()
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize observation normalization
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        
        # Load ARS policy if provided
        ars_weights = None
        obs_filter = None
        if ars_policy_path and ars_params_path:
            ars_weights, mu, std, _ = load_ars_policy(ars_policy_path, ars_params_path)
            obs_filter = {'mu': torch.FloatTensor(mu), 'std': torch.FloatTensor(std)}
        
        # Initialize networks
        self.policy = GaussianPolicy(self.obs_dim, self.action_dim, obs_filter=obs_filter).to(self.device)
        self.q1 = QNetwork(self.obs_dim, self.action_dim, obs_filter=obs_filter).to(self.device)
        self.q2 = QNetwork(self.obs_dim, self.action_dim, obs_filter=obs_filter).to(self.device)
        self.target_q1 = QNetwork(self.obs_dim, self.action_dim, obs_filter=obs_filter).to(self.device)
        self.target_q2 = QNetwork(self.obs_dim, self.action_dim, obs_filter=obs_filter).to(self.device)
        
        # Copy parameters to target networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.action_dim, buffer_size)
        
        # Initialize entropy tuning
        if self.auto_entropy:
            if target_entropy is None:
                self.target_entropy = -np.prod(self.action_dim).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        # Initialize workers
        self.workers = [SACWorker.remote(i, obs_filter=obs_filter) for i in range(num_workers)]
        
        # Setup logging
        if logdir:
            logz.configure_output_dir(logdir)
            logz.save_params(params)
            
        # Track best policy
        self.best_mean_reward = float('-inf')
        
        # Initialize metrics tracking
        self.metrics = {
            'policy_loss': [],
            'q1_loss': [],
            'q2_loss': [],
            'alpha_loss': [],
            'alpha': [],
            'mean_reward': [],
            'std_reward': [],
            'max_reward': [],
            'min_reward': [],
            'timesteps': []
        }
    
    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        
        # Update observation normalization
        self.obs_rms.update(obs.cpu().numpy())
        
        # Update Q-functions
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_obs)
            target_q1 = self.target_q1(next_obs, next_actions)
            target_q2 = self.target_q2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_log_probs)
        
        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions, log_probs = self.policy.sample(obs)
        q1 = self.q1(obs, new_actions)
        q2 = self.q2(obs, new_actions)
        q = torch.min(q1, q2)
        
        policy_loss = (self.alpha * log_probs - q).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha
        alpha_loss = None
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Update target networks
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            'policy_loss': policy_loss.item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0,
            'alpha': self.alpha
        }
    
    def train(self, num_iterations):
        start_time = time.time()
        total_timesteps = 0
        
        # Initial data collection
        print("Collecting initial data...")
        for _ in range(1000):  # Collect some initial data
            rollout_ids = [worker.collect_rollout.remote() for worker in self.workers]
            rollouts = ray.get(rollout_ids)
            
            for rollout in rollouts:
                obs = rollout['observations'].cpu().numpy()
                actions = rollout['actions'].cpu().numpy()
                rewards = rollout['rewards'].cpu().numpy()
                next_obs = rollout['next_observations'].cpu().numpy()
                dones = rollout['dones'].cpu().numpy()
                
                for i in range(len(obs)):
                    self.replay_buffer.add(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
        
        print("Starting training...")
        for i in range(num_iterations):
            # Collect rollouts from all workers
            rollout_ids = [worker.collect_rollout.remote() for worker in self.workers]
            rollouts = ray.get(rollout_ids)
            
            # Add to replay buffer
            for rollout in rollouts:
                obs = rollout['observations'].cpu().numpy()
                actions = rollout['actions'].cpu().numpy()
                rewards = rollout['rewards'].cpu().numpy()
                next_obs = rollout['next_observations'].cpu().numpy()
                dones = rollout['dones'].cpu().numpy()
                
                for j in range(len(obs)):
                    self.replay_buffer.add(obs[j], actions[j], rewards[j], next_obs[j], dones[j])
            
            # Update policy
            for _ in range(100):  # Multiple updates per iteration
                if self.replay_buffer.size > self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    metrics = self.update(batch)
            
            # Evaluate current policy
            eval_rollouts = ray.get([worker.collect_rollout.remote() for worker in self.workers[:5]])
            eval_rewards = [rollout['rewards'].sum().item() for rollout in eval_rollouts]
            
            # Update metrics
            self.metrics['policy_loss'].append(metrics['policy_loss'])
            self.metrics['q1_loss'].append(metrics['q1_loss'])
            self.metrics['q2_loss'].append(metrics['q2_loss'])
            self.metrics['alpha_loss'].append(metrics['alpha_loss'])
            self.metrics['alpha'].append(metrics['alpha'])
            self.metrics['mean_reward'].append(np.mean(eval_rewards))
            self.metrics['std_reward'].append(np.std(eval_rewards))
            self.metrics['max_reward'].append(np.max(eval_rewards))
            self.metrics['min_reward'].append(np.min(eval_rewards))
            
            # Update timesteps
            total_timesteps += sum(len(rollout['rewards']) for rollout in rollouts)
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
                logz.log_tabular("Q1Loss", metrics['q1_loss'])
                logz.log_tabular("Q2Loss", metrics['q2_loss'])
                logz.log_tabular("AlphaLoss", metrics['alpha_loss'])
                logz.log_tabular("Alpha", metrics['alpha'])
                logz.dump_tabular()
                
                # Save best policy
                if np.mean(eval_rewards) > self.best_mean_reward:
                    self.best_mean_reward = np.mean(eval_rewards)
                    if self.logdir:
                        torch.save({
                            'policy_state_dict': self.policy.state_dict(),
                            'q1_state_dict': self.q1.state_dict(),
                            'q2_state_dict': self.q2.state_dict(),
                            'target_q1_state_dict': self.target_q1.state_dict(),
                            'target_q2_state_dict': self.target_q2.state_dict(),
                            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
                            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
                            'obs_rms': self.obs_rms,
                            'best_reward': self.best_mean_reward
                        }, os.path.join(self.logdir, 'best_policy.pt'))
                        
                # Save metrics for plotting
                np.save(os.path.join(self.logdir, 'metrics.npy'), self.metrics)

def run_sac(params):
    """Run SAC training with the given parameters."""
    # Create log directory
    logdir = os.path.join('data', 'sac_' + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    
    # Initialize trainer
    trainer = SACTrainer(
        num_workers=params.get('num_workers', 64),
        batch_size=params.get('batch_size', 256),
        buffer_size=params.get('buffer_size', int(1e6)),
        learning_rate=params.get('learning_rate', 3e-4),
        gamma=params.get('gamma', 0.99),
        tau=params.get('tau', 0.005),
        alpha=params.get('alpha', 0.2),
        auto_entropy=params.get('auto_entropy', True),
        target_entropy=params.get('target_entropy', None),
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
    parser.add_argument('--num_workers', type=int, default=64,
                      help='Number of parallel workers')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for updates')
    parser.add_argument('--buffer_size', type=int, default=int(1e6),
                      help='Size of replay buffer')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                      help='Target network update rate')
    parser.add_argument('--alpha', type=float, default=0.2,
                      help='Initial entropy coefficient')
    parser.add_argument('--auto_entropy', action='store_true',
                      help='Whether to automatically tune entropy coefficient')
    parser.add_argument('--target_entropy', type=float, default=None,
                      help='Target entropy for automatic tuning')
    parser.add_argument('--num_iterations', type=int, default=1000,
                      help='Number of training iterations')
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init()
    
    # Set up parameters
    params = vars(args)
    
    # Run training
    logdir = run_sac(params)
    print(f"Training complete. Logs saved to {logdir}") 