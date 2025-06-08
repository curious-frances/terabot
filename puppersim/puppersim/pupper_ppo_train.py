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
from torch.optim.lr_scheduler import CosineAnnealingLR

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

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128], obs_filter=None):
        super(ActorCritic, self).__init__()
        
        # Actor (Policy) Network
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        actor_layers.append(nn.Tanh())  # Bound actions to [-1, 1]
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic (Value) Network
        critic_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Learnable standard deviation for action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Observation filter
        self.obs_filter = obs_filter
        
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
                 num_workers=64,
                 num_epochs=20,
                 batch_size=256,
                 clip_ratio=0.1,
                 value_coef=0.5,
                 entropy_coef=0.05,
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
        
        # Initialize observation normalization
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        
        # Initialize policy
        self.policy = ActorCritic(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # Initialize workers
        self.workers = [PPOWorker.remote(i) for i in range(num_workers)]
        
        # Setup logging
        if logdir:
            logz.configure_output_dir(logdir)
            logz.save_params(params)
            
        # Track best policy
        self.best_mean_reward = float('-inf')
        
        # Enhanced metrics tracking
        self.metrics = {
            # Training metrics
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'learning_rate': [],
            'gradient_norm': [],
            'clip_fraction': [],
            
            # Performance metrics
            'mean_reward': [],
            'std_reward': [],
            'max_reward': [],
            'min_reward': [],
            'episode_length': [],
            'success_rate': [],
            
            # Policy metrics
            'action_mean': [],
            'action_std': [],
            'value_mean': [],
            'value_std': [],
            
            # Stability metrics
            'height_error': [],
            'pitch_error': [],
            'roll_error': [],
            'joint_velocity': [],
            
            # Efficiency metrics
            'energy_consumption': [],
            'torque_usage': [],
            
            # Timing metrics
            'timesteps': [],
            'wall_time': [],
            'update_time': [],
            'rollout_time': []
        }
        
        # Initialize time tracking
        self.start_time = time.time()
        self.last_update_time = self.start_time
    
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
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, rollouts):
        update_start_time = time.time()
        
        obs = rollouts['observations']
        actions = rollouts['actions']
        old_values = rollouts['values']
        old_log_probs = rollouts['log_probs']
        rewards = rollouts['rewards']
        
        # Update observation normalization
        self.obs_rms.update(obs.cpu().numpy())
        
        # Compute advantages
        with torch.no_grad():
            next_value = self.policy.critic(obs[-1].unsqueeze(0)).squeeze()
        advantages, returns = self.compute_advantages(rewards, old_values, next_value)
        
        # Track policy statistics
        self.metrics['action_mean'].append(actions.mean().item())
        self.metrics['action_std'].append(actions.std().item())
        self.metrics['value_mean'].append(old_values.mean().item())
        self.metrics['value_std'].append(old_values.std().item())
        
        # Update policy for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        num_updates = 0
        
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
                
                # Track clip fraction
                clip_fraction = (ratio > (1 + self.clip_ratio)).float().mean() + \
                               (ratio < (1 - self.clip_ratio)).float().mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                
                # Track gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.metrics['gradient_norm'].append(grad_norm.item())
                
                self.optimizer.step()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_fraction += clip_fraction.item()
                num_updates += 1
        
        # Update learning rate
        self.scheduler.step()
        
        # Compute average metrics
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        avg_clip_fraction = total_clip_fraction / num_updates
        
        # Track timing
        update_time = time.time() - update_start_time
        self.metrics['update_time'].append(update_time)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'clip_fraction': avg_clip_fraction
        }
    
    def train(self, num_iterations):
        start_time = time.time()
        total_timesteps = 0
        
        for i in range(num_iterations):
            rollout_start_time = time.time()
            
            # Collect rollouts from all workers
            rollout_ids = [worker.collect_rollout.remote() for worker in self.workers]
            rollouts = ray.get(rollout_ids)
            
            # Track rollout time
            rollout_time = time.time() - rollout_start_time
            self.metrics['rollout_time'].append(rollout_time)
            
            # Update policy
            metrics = self.update_policy(rollouts[0])
            
            # Evaluate current policy
            eval_rollouts = ray.get([worker.collect_rollout.remote() for worker in self.workers[:5]])
            
            # Compute detailed evaluation metrics
            eval_rewards = []
            eval_lengths = []
            eval_successes = []
            eval_heights = []
            eval_pitches = []
            eval_rolls = []
            eval_joint_velocities = []
            eval_energies = []
            
            for rollout in eval_rollouts:
                rewards = rollout['rewards']
                observations = rollout['observations']
                actions = rollout['actions']
                
                eval_rewards.append(rewards.sum().item())
                eval_lengths.append(len(rewards))
                eval_successes.append(rewards[-1] > 0)  # Success if final reward is positive
                
                # Extract stability metrics
                heights = observations[:, 1]  # Height
                pitches = observations[:, 2]  # Pitch
                rolls = observations[:, 3]    # Roll
                
                eval_heights.append(heights.mean().item())
                eval_pitches.append(pitches.mean().item())
                eval_rolls.append(rolls.mean().item())
                
                # Compute joint velocities
                joint_velocities = torch.diff(observations[:, 4:16], dim=0)
                eval_joint_velocities.append(joint_velocities.abs().mean().item())
                
                # Compute energy consumption
                energy = torch.sum(actions ** 2, dim=1).mean().item()
                eval_energies.append(energy)
            
            # Update metrics
            self.metrics['policy_loss'].append(metrics['policy_loss'])
            self.metrics['value_loss'].append(metrics['value_loss'])
            self.metrics['entropy'].append(metrics['entropy'])
            self.metrics['clip_fraction'].append(metrics['clip_fraction'])
            self.metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            self.metrics['mean_reward'].append(np.mean(eval_rewards))
            self.metrics['std_reward'].append(np.std(eval_rewards))
            self.metrics['max_reward'].append(np.max(eval_rewards))
            self.metrics['min_reward'].append(np.min(eval_rewards))
            self.metrics['episode_length'].append(np.mean(eval_lengths))
            self.metrics['success_rate'].append(np.mean(eval_successes))
            
            self.metrics['height_error'].append(np.mean([abs(h - 0.15) for h in eval_heights]))
            self.metrics['pitch_error'].append(np.mean([abs(p) for p in eval_pitches]))
            self.metrics['roll_error'].append(np.mean([abs(r) for r in eval_rolls]))
            self.metrics['joint_velocity'].append(np.mean(eval_joint_velocities))
            self.metrics['energy_consumption'].append(np.mean(eval_energies))
            
            # Update timesteps and timing
            total_timesteps += len(rollouts[0]['rewards'])
            self.metrics['timesteps'].append(total_timesteps)
            self.metrics['wall_time'].append(time.time() - start_time)
            
            # Log metrics
            if (i + 1) % 10 == 0:
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("Time", time.time() - start_time)
                logz.log_tabular("Timesteps", total_timesteps)
                logz.log_tabular("MeanReward", np.mean(eval_rewards))
                logz.log_tabular("StdReward", np.std(eval_rewards))
                logz.log_tabular("MaxReward", np.max(eval_rewards))
                logz.log_tabular("MinReward", np.min(eval_rewards))
                logz.log_tabular("SuccessRate", np.mean(eval_successes))
                logz.log_tabular("EpisodeLength", np.mean(eval_lengths))
                logz.log_tabular("PolicyLoss", metrics['policy_loss'])
                logz.log_tabular("ValueLoss", metrics['value_loss'])
                logz.log_tabular("Entropy", metrics['entropy'])
                logz.log_tabular("ClipFraction", metrics['clip_fraction'])
                logz.log_tabular("GradientNorm", np.mean(self.metrics['gradient_norm'][-10:]))
                logz.log_tabular("HeightError", np.mean([abs(h - 0.15) for h in eval_heights]))
                logz.log_tabular("EnergyConsumption", np.mean(eval_energies))
                logz.dump_tabular()
                
                # Save best policy
                if np.mean(eval_rewards) > self.best_mean_reward:
                    self.best_mean_reward = np.mean(eval_rewards)
                    if self.logdir:
                        torch.save({
                            'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'obs_rms': self.obs_rms,
                            'best_reward': self.best_mean_reward,
                            'metrics': self.metrics
                        }, os.path.join(self.logdir, 'best_policy.pt'))
                        
                # Save metrics for plotting
                np.save(os.path.join(self.logdir, 'metrics.npy'), self.metrics)
                
                # Save detailed metrics to CSV
                metrics_df = pd.DataFrame(self.metrics)
                metrics_df.to_csv(os.path.join(self.logdir, 'detailed_metrics.csv'))

def run_ppo(params):
    """Run PPO training with the given parameters."""
    # Create log directory
    logdir = os.path.join('data', 'ppo_' + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    
    # Initialize trainer
    trainer = PPOTrainer(
        num_workers=params.get('num_workers', 64),
        num_epochs=params.get('num_epochs', 20),
        batch_size=params.get('batch_size', 256),
        clip_ratio=params.get('clip_ratio', 0.1),
        value_coef=params.get('value_coef', 0.5),
        entropy_coef=params.get('entropy_coef', 0.05),
        learning_rate=params.get('learning_rate', 3e-4),
        gamma=params.get('gamma', 0.99),
        lam=params.get('lam', 0.95),
        logdir=logdir,
        params=params
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
    parser.add_argument('--num_epochs', type=int, default=20,
                      help='Number of epochs per update')
    parser.add_argument('--batch_size', type=int, default=256,
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