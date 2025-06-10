import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gym
import time
import os
import json
import argparse
from puppersim.reacher import reacher_env
import arspb.logz as logz

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Action std
        self.action_std = nn.Parameter(torch.ones(action_dim) * 0.5)
        self.action_scale = torch.tensor(2 * np.pi)  # Based on action space bounds
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        action_std = self.action_std.expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action = dist.rsample()
        action = torch.tanh(action) * self.action_scale
        action_logprob = dist.log_prob(action)
        action_logprob -= torch.log(self.action_scale * (1 - torch.tanh(action).pow(2)) + 1e-6)
        action_logprob = action_logprob.sum(1, keepdim=True)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_std = self.action_std.expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action)
        action_logprobs -= torch.log(self.action_scale * (1 - torch.tanh(action).pow(2)) + 1e-6)
        action_logprobs = action_logprobs.sum(1, keepdim=True)
        dist_entropy = dist.entropy().mean()
        
        state_value = self.critic(state)
        
        return action_logprobs, state_value, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1))
            action, action_logprob = self.policy_old.act(state)
        
        return action.cpu().numpy()[0], action_logprob.cpu().numpy()[0]
    
    def update(self, memory):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert list to tensor
        old_states = torch.FloatTensor(np.array(memory.states))
        old_actions = torch.FloatTensor(np.array(memory.actions))
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs)).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return loss.mean().item()

def train_ppo(params):
    env = reacher_env.ReacherEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    ppo = PPO(state_dim, action_dim)
    memory = Memory()
    
    total_timesteps = 0
    episode_rewards = []
    episode_lengths = []
    policy_losses = []
    
    for episode in range(params['n_iter']):
        episode_reward = 0
        episode_length = 0
        state = env.reset()
        done = False
        
        while not done:
            # Running policy_old
            action, action_logprob = ppo.select_action(state)
            
            # Performing action
            next_state, reward, done, _ = env.step(action)
            
            # Storing the transition
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_timesteps += 1
            
            # Update if it's time
            if len(memory.states) >= 2048:  # Update every 2048 steps
                policy_loss = ppo.update(memory)
                policy_losses.append(policy_loss)
                memory.clear_memory()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 10 == 0:
            # Log statistics
            logz.log_tabular("Time", time.time() - params['start_time'])
            logz.log_tabular("Iteration", episode + 1)
            logz.log_tabular("AverageReward", np.mean(episode_rewards[-10:]))
            logz.log_tabular("StdRewards", np.std(episode_rewards[-10:]))
            logz.log_tabular("MaxReward", np.max(episode_rewards[-10:]))
            logz.log_tabular("MinReward", np.min(episode_rewards[-10:]))
            logz.log_tabular("AverageLength", np.mean(episode_lengths[-10:]))
            logz.log_tabular("Timesteps", total_timesteps)
            if len(policy_losses) > 0:
                logz.log_tabular("PolicyLoss", np.mean(policy_losses[-100:]))
            logz.dump_tabular()
            
            # Save model
            if not os.path.exists(params['dir_path']):
                os.makedirs(params['dir_path'])
            torch.save({
                'policy_state_dict': ppo.policy.state_dict(),
                'policy_old_state_dict': ppo.policy_old.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
            }, os.path.join(params['dir_path'], f'ppo_model_{episode+1}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', '-n', type=int, default=50000)
    parser.add_argument('--dir_path', type=str, default='data/ppo')
    parser.add_argument('--seed', type=int, default=37)
    args = parser.parse_args()
    params = vars(args)
    params['start_time'] = time.time()
    
    # Set random seeds
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    
    # Configure logging
    logz.configure_output_dir(params['dir_path'])
    logz.save_params(params)
    
    train_ppo(params) 