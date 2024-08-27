"""
Author: Jeff Asante
Date: 27 August 2024, 3:37 AM GMT
GitHub: https://github.com/jeffasante/RL-PPO-SharpShooter

Description: This script implements the training loop for a PPO-based agent in a Shooter environment.
The agent is trained over multiple episodes with the ability to save models periodically and 
plot training metrics. Pygame is used for rendering and managing events during the training process.
"""

import matplotlib.pyplot as plt
import os
import time

import pygame

# Import your ShooterEnv
from shooter_env import ShooterEnv
from ppo_agent import PPO

# Training loop 
def train(env, ppo_agent, num_episodes, save_freq=100):
    """
    Main training loop for the PPO agent.

    Args:
    - env: The environment in which the agent is trained.
    - ppo_agent: The PPO agent being trained.
    - num_episodes: Total number of episodes to train.
    - save_freq: Frequency (in episodes) to save the model.
   """
    os.makedirs('models', exist_ok=True)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        
        while not done:
            action, log_prob = ppo_agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
            
            # Render the game
            env.render()
            
            # Add a small delay to make the game visible
            time.sleep(0.05)
            
            # Handle Pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        ppo_agent.update(states, actions, log_probs, rewards, dones)
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        
        # Save model periodically
        if (episode + 1) % save_freq == 0:
            ppo_agent.save_model(f'models/ppo_model_episode_{episode+1}.pth')
            ppo_agent.plot_losses()
    
    # Plot episode rewards
    plt.figure(figsize=(12, 8))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('episode_rewards.png')
    plt.close()


# Main execution
# Main executionx
if __name__ == "__main__":
    # env = ShooterEnv()  # Create the environment
    env = ShooterEnv(speed_multiplier=2, skip_frames=2) # Add desired speed and frame skip settings
    
    state_dim = (1, 84, 84)  # Input state dimension (channels, height, width)
    action_dim = env.action_space.n  # Number of possible actions
    
    # Initialize PPO agent with hyperparameters
    ppo_agent = PPO(state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01)
    
    # Train the agent
    train(env, ppo_agent, num_episodes=1000, save_freq=100)
    
    # After training, save the final model and plot losses
    ppo_agent.save_model('models/ppo_model_final.pth')
    ppo_agent.plot_losses()
    
    pygame.quit()