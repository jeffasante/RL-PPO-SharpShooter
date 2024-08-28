"""
Author: Jeff Asante
Date: 27 August 2024, 3:37 AM GMT
GitHub: https://github.com/jeffasante/RL-PPO-SharpShooter

Description: This script implements the Proximal Policy Optimization (PPO) algorithm using 
a neural network-based Actor-Critic model for training in a custom Shooter environment. 
The script supports both CPU and Apple Silicon GPUs (MPS) for computation. It includes 
the core PPO algorithm, neural network definitions, and a training loop that periodically 
saves the model and plots training losses.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
import time
import pygame

# Check if MPS is available (Apple Silicon GPUs)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")


# Neural Network for both Actor and Critic
class ActorCritic(nn.Module):
    """
    A neural network model for PPO that shares the first few layers between 
    the Actor and Critic networks. The Actor network predicts the action 
    probabilities, and the Critic network predicts the value of the current state.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initializes the Actor-Critic network.

        Args:
        - state_dim: The dimension of the input state (height, width, channels).
        - action_dim: The number of possible actions in the environment.
        """
        super(ActorCritic, self).__init__()
        
        # Shared layers between actor and critic (Convolutional layers)
        self.shared = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)
        
        # Calculate the size of the flattened output after convolutions
        self.flat_size = self._get_conv_output(state_dim)
        
        # Actor network for policy prediction
        self.actor = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim), # Output probabilities for each action
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Critic network for state-value prediction
        self.critic = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # Output a single value
        ).to(device)
        
    def _get_conv_output(self, shape):
        """
        Helper function to determine the size of the output after passing through the convolutional layers.

        Args:
        - shape: The shape of the input state (height, width, channels).
        
        Returns:
        - The size of the flattened output from the convolutional layers.
        """
        bs = 1
        input = torch.rand(bs, *shape).to(device)
        output = self.shared(input)
        return int(np.prod(output.size()))
        
    def forward(self, state):
        """
        Forward pass through the network to get action probabilities and state value.

        Args:
        - state: The input state tensor.

        Returns:
        - action_probs: The probabilities of each action.
        - state_value: The value of the current state.
        """
        x = self.shared(state)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
    

class PPO:
    """
    Proximal Policy Optimization (PPO) agent implementation for training with 
    Actor-Critic models.
    """
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, value_coef, entropy_coef):
        """
        Initializes the PPO agent.

        Args:
        - state_dim: Dimension of the input state.
        - action_dim: Number of possible actions.
        - lr: Learning rate.
        - gamma: Discount factor for rewards.
        - epsilon: Clipping parameter for PPO.
        - value_coef: Coefficient for value loss.
        - entropy_coef: Coefficient for entropy loss.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
       
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        
    def get_action(self, state):
        """
        Selects an action based on the current state using the Actor network.

        Args:
        - state: The current state of the environment.

        Returns:
        - action: The selected action.
        - log_prob: The log probability of the selected action.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension and move to device
        state = state.permute(0, 3, 1, 2)  # Change to [batch, channel, height, width]
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, states, actions, old_log_probs, rewards, dones):
        """
        Updates the Actor-Critic networks using PPO loss.

        Args:
        - states: List of states.
        - actions: List of actions taken.
        - old_log_probs: Log probabilities of actions taken (from previous policy).
        - rewards: List of rewards received.
        - dones: List indicating whether the episode ended.
        """
        states = torch.FloatTensor(np.array(states)).to(device)
        states = states.permute(0, 3, 1, 2)  # Change to [batch, channel, height, width]
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Compute advantages
        with torch.no_grad():
            _, state_values = self.actor_critic(states)
            state_values = state_values.squeeze()
        
        advantages = torch.zeros_like(rewards).to(device)
        returns = torch.zeros_like(rewards).to(device)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
            advantages[t] = running_return - state_values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(5):  # Run multiple epochs
            action_probs, state_values = self.actor_critic(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # Compute the ratio (new policy / old policy)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Compute critic loss as MSE between predicted value and actual return
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Store losses for plotting
            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())
            self.entropy_losses.append(entropy.item())
            
            
    def save_model(self, path):
        """
        Saves the model weights to a file.

        Args:
        - path: The file path to save the model.
        """
        torch.save(self.actor_critic.state_dict(), path)
    
    def load_model(self, path):
        """
        Loads the model weights from a file.

        Args:
        - path: The file path from which to load the model.
        """
        self.actor_critic.load_state_dict(torch.load(path, map_location=device))

    def plot_losses(self):
        """
        Plots and saves the loss curves for Actor, Critic, and Entropy losses.
        """
        plt.figure(figsize=(12, 8))
        plt.plot(self.actor_losses, label='Actor Loss')
        plt.plot(self.critic_losses, label='Critic Loss')
        plt.plot(self.entropy_losses, label='Entropy Loss')
        plt.title('PPO Losses')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('ppo_losses.png')
        plt.close()
