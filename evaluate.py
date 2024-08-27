"""
Author: Jeff Asante
Date: 27 August 2024, 3:37 AM GMT
GitHub: https://github.com/jeffasante/RL-PPO-SharpShooter

Description: This script implements the main execution for a PPO-based agent in a Shooter environment.
The agent is initialized with specific hyperparameters and performs inference using a pre-trained model.
Pygame is used for rendering and managing events during the inference process.
"""


import time

import pygame

from shooter_env import ShooterEnv
from ppo_agent import PPO
import os

# Main execution
if __name__ == "__main__":
    env = ShooterEnv(speed_multiplier=2, skip_frames=2)  # Create the environment with desired settings
    
    state_dim = (1, 84, 84)  # Input state dimension (channels, height, width)
    action_dim = env.action_space.n  # Number of possible actions
    
    # Initialize PPO agent with hyperparameters
    ppo_agent = PPO(state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01)
    
    # Load a saved model for inference
    # model_path = 'models/ppo_model_final.pth'  # Specify the path to the saved model
    model_path = 'models/ppo_model_episode_300.pth'
    
    if os.path.exists(model_path):
        ppo_agent.load_model(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"No saved model found at {model_path}, cannot proceed with inference.")
        exit()

    # Run inference
    state = env.reset()
    done = False
    while not done:
        action, _ = ppo_agent.get_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.05)
        
        # Handle Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    
    pygame.quit()
