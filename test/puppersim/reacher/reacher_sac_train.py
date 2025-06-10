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

class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, 3))  # target position
        self.action = np.zeros((max_size, 3))  # joint angles
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, 3))
        self.done = np.zeros((max_size, 1))
        
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.done[ind])
        )

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.action_scale = torch.tensor(2 * np.pi)  # Based on action space bounds
        self.action_bias = torch.tensor(0.0)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
        
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)
        
        self.target_q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q2 = QNetwork(state_dim, action_dim, hidden_dim)
        
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        self.replay_buffer = ReplayBuffer()
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def train(self, batch_size=256):
        if self.replay_buffer.size < batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        # Update Q-functions
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q = reward + (1 - done) * self.gamma * target_q
            
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()
        
        # Update policy
        new_action, log_pi, _ = self.policy.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_pi - q_new).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        # Update target networks
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return q1_loss.item(), q2_loss.item(), policy_loss.item()

def train_sac(params):
    env = reacher_env.ReacherEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SAC(state_dim, action_dim, hidden_dim=256)
    
    total_timesteps = 0
    episode_rewards = []
    episode_lengths = []
    q1_losses = []
    q2_losses = []
    policy_losses = []
    
    for episode in range(params['n_iter']):
        episode_reward = 0
        episode_length = 0
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_timesteps += 1
            
            if total_timesteps > 1000:  # Start training after collecting some samples
                q1_loss, q2_loss, policy_loss = agent.train()
                if q1_loss is not None:
                    q1_losses.append(q1_loss)
                    q2_losses.append(q2_loss)
                    policy_losses.append(policy_loss)
        
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
            if len(q1_losses) > 0:
                logz.log_tabular("Q1Loss", np.mean(q1_losses[-100:]))
                logz.log_tabular("Q2Loss", np.mean(q2_losses[-100:]))
                logz.log_tabular("PolicyLoss", np.mean(policy_losses[-100:]))
            logz.dump_tabular()
            
            # Save model
            if not os.path.exists(params['dir_path']):
                os.makedirs(params['dir_path'])
            torch.save({
                'policy_state_dict': agent.policy.state_dict(),
                'q1_state_dict': agent.q1.state_dict(),
                'q2_state_dict': agent.q2.state_dict(),
                'policy_optimizer_state_dict': agent.policy_optim.state_dict(),
                'q1_optimizer_state_dict': agent.q1_optim.state_dict(),
                'q2_optimizer_state_dict': agent.q2_optim.state_dict(),
            }, os.path.join(params['dir_path'], f'sac_model_{episode+1}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', '-n', type=int, default=50000)
    parser.add_argument('--dir_path', type=str, default='data/sac')
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
    
    train_sac(params) 