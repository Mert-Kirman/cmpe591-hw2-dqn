import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

from homework2 import Hw2Env


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# Q-Network (MLP for High-Level State)
class QNetwork(nn.Module):
    def __init__(self, state_dim=6, n_actions=8):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim=6, n_actions=8):
        self.n_actions = n_actions
        
        self.device = torch.device("cpu") # Training was faster on CPU
        print(f"Training on device: {self.device}")
        
        # Hyperparameters
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 10000
        self.tau = 0.005
        self.batch_size = 128
        self.learning_rate = 0.0001
        self.update_freq = 4  # (e.g., Update network every 4 steps)
        self.memory = ReplayBuffer(10000)
        
        self.steps_done = 0
        
        # Networks
        self.online_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() 
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss() 
        
    def select_action(self, state):
        # Continuous exponential decay for smooth transition from pure exploration to almost pure exploitation
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)    # Shape: (1, state_dim)
            return self.online_net(state_tensor).argmax().item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)  # Shape: (batch_size, state_dim)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)    # Shape: (batch_size, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)   # Shape: (batch_size, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)  # Shape: (batch_size, state_dim)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # Shape: (batch_size, 1)

        # Current Q values
        q_values = self.online_net(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
        # Compute loss and optimize
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents explosive updates
        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), 100) 
        self.optimizer.step()
        
        # Soft update the target network
        self.soft_update()

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        online_net_state_dict = self.online_net.state_dict()
        for key in online_net_state_dict:
            target_net_state_dict[key] = online_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

if __name__ == "__main__":
    env = Hw2Env(n_actions=8, render_mode="blind")
    agent = DQNAgent(state_dim=6, n_actions=8)
    
    num_episodes = 2500
    episode_rewards = []
    episode_rps = [] # Reward Per Step
    global_steps = 0  # Track total steps for update frequency
    
    for episode in tqdm(range(num_episodes), desc="Training DQN"):
        env.reset()
        state = env.high_level_state()
        
        cumulative_reward = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            _, reward, terminal, truncated = env.step(action)
            done = terminal or truncated
            
            next_state = env.high_level_state()
            
            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            cumulative_reward += reward
            episode_steps += 1
            global_steps += 1
            
            # Perform optimization every update_freq steps
            if global_steps % agent.update_freq == 0:
                agent.optimize_model()
            
        episode_rewards.append(cumulative_reward)
        episode_rps.append(cumulative_reward / max(episode_steps, 1))
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rps = np.mean(episode_rps[-100:])
            tqdm.write(f"Episode {episode+1} | Avg Reward (last 100): {avg_reward:.2f} | Avg RPS: {avg_rps:.2f}")

    # Save model weights
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "dqn_model.pt")
    torch.save({
        'online_net_state_dict': agent.online_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode': num_episodes,
        'steps_done': agent.steps_done
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Plot Results
    figure_dir = "assets"
    os.makedirs(figure_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, "dqn_training_results.png")
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Cumulative Reward
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6, color='blue')
    # Add a moving average for cleaner visualization
    moving_avg_reward = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
    plt.plot(moving_avg_reward, color='red', label='50-Episode Moving Avg')
    plt.title('Cumulative Reward over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot 2: Reward Per Step (RPS)
    plt.subplot(1, 2, 2)
    plt.plot(episode_rps, alpha=0.6, color='green')
    moving_avg_rps = np.convolve(episode_rps, np.ones(50)/50, mode='valid')
    plt.plot(moving_avg_rps, color='darkgreen', label='50-Episode Moving Avg')
    plt.title('Reward Per Step (RPS) over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('RPS')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(figure_path)
    print(f"Training complete! Plot saved to {figure_path}")
