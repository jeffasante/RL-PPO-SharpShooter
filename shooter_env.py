"""
Author: Jeff Asante
Date: 27 August 2024, 3:37 AM GMT
GitHub: https://github.com/jeffasante/RL-PPO-SharpShooter

Description: This script defines the Shooter environment for training and evaluating a PPO-based agent.
The environment is built using Pygame and Gym, featuring player and CPU enemy interactions, movement, 
shooting mechanics, and health-based rewards. The environment includes adjustable speed and frame skip settings 
to control the difficulty and pacing of the game.
"""

import gym
import pygame
import numpy as np
import random
from gym import spaces

class ShooterEnv(gym.Env):
    def __init__(self, speed_multiplier=2, skip_frames=2):
        super().__init__()
        
        # Initialize Pygame and game properties
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Speed and frame skip settings
        self.speed_multiplier = speed_multiplier
        self.skip_frames = skip_frames

        # Adjust speed-related parameters
        self.bullet_speed = 7 * speed_multiplier
        self.player_move_speed = 5 * speed_multiplier
        self.cpu_move_speed = 3 * speed_multiplier
        self.player_shoot_delay = max(1, 20 // speed_multiplier)
        self.cpu_shoot_delay = max(1, 50 // speed_multiplier)

        # Player properties
        self.player_size = 50
        self.player_x = self.width // 2
        self.player_y = self.height - self.player_size
        self.player_hp = 100
        self.bullets = []
        self.player_shoot_counter = 0

        # CPU enemy properties
        self.cpu_size = 50
        self.cpu_x = random.randint(0, self.width - self.cpu_size)
        self.cpu_y = 60  # Position the CPU below the scoreboard
        self.cpu_hp = 100
        self.cpu_bullets = []
        self.cpu_shoot_counter = 0
        self.cpu_direction = random.choice([-1, 1])

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Left, Right, Shoot
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)


    def step(self, action):
        total_reward = 0
        done = False
        for _ in range(self.skip_frames):
            observation, reward, done, info = self._step(action)
            total_reward += reward
            if done:
                break
        return observation, total_reward, done, info

    def _step(self, action):
        # Apply action
        if action == 0:  # Move left
            self.player_x = max(0, self.player_x - self.player_move_speed)
        elif action == 1:  # Move right
            self.player_x = min(self.width - self.player_size, self.player_x + self.player_move_speed)
        elif action == 2:  # Shoot
            self.player_shoot_counter += 1
            if self.player_shoot_counter >= self.player_shoot_delay:
                self.bullets.append(pygame.Rect(self.player_x + self.player_size // 2, self.player_y, 5, 10))
                self.player_shoot_counter = 0

        # Update game state
        self._update_game_state()

        # Check for game over
        done = self.player_hp <= 0 or self.cpu_hp <= 0

        # Calculate reward
        reward = self._calculate_reward()

        # Get observation
        observation = self._get_observation()

        return observation, reward, done, {}

    def reset(self):
        # Reset game state
        self.player_x = self.width // 2
        self.cpu_x = random.randint(0, self.width - self.cpu_size)
        self.player_hp = 100
        self.cpu_hp = 100
        self.bullets = []
        self.cpu_bullets = []
        self.cpu_direction = random.choice([-1, 1])
        return self._get_observation()


    def render(self, mode='human'):
        # Render the game state
        self.screen.fill((0, 0, 0))
        
        # Render the scoreboard first, so it appears on top
        self._render_scoreboard()
        
        # Render the player and CPU beneath the scoreboard
        pygame.draw.rect(self.screen, (0, 0, 255), (self.player_x, self.player_y, self.player_size, self.player_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.cpu_x, self.cpu_y, self.cpu_size, self.cpu_size))
        
        for bullet in self.bullets:
            pygame.draw.rect(self.screen, (255, 255, 255), bullet)
        for bullet in self.cpu_bullets:
            pygame.draw.rect(self.screen, (255, 255, 0), bullet)
        
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.fps * self.speed_multiplier)
        elif mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


    def _update_game_state(self):
        # Move bullets
        self.bullets = [bullet.move(0, -self.bullet_speed) for bullet in self.bullets if bullet.y > 0]
        self.cpu_bullets = [bullet.move(0, self.bullet_speed) for bullet in self.cpu_bullets if bullet.y < self.height]
        
        # Move CPU
        self._move_cpu()
        
        # Check collisions
        self._check_collisions()

    def _move_cpu(self):
        if random.random() < 0.01 * self.speed_multiplier:
            self.cpu_direction = -self.cpu_direction

        self.cpu_x += self.cpu_direction * self.cpu_move_speed
        self.cpu_x = max(0, min(self.width - self.cpu_size, self.cpu_x))

        self.cpu_shoot_counter += 1
        if self.cpu_shoot_counter >= self.cpu_shoot_delay:
            self.cpu_bullets.append(pygame.Rect(self.cpu_x + self.cpu_size // 2, self.cpu_y + self.cpu_size, 5, 10))
            self.cpu_shoot_counter = 0

    def _check_collisions(self):
        for bullet in self.bullets[:]:
            if bullet.colliderect(pygame.Rect(self.cpu_x, self.cpu_y, self.cpu_size, self.cpu_size)):
                self.bullets.remove(bullet)
                self.cpu_hp -= 10

        for bullet in self.cpu_bullets[:]:
            if bullet.colliderect(pygame.Rect(self.player_x, self.player_y, self.player_size, self.player_size)):
                self.cpu_bullets.remove(bullet)
                self.player_hp -= 10

    def _calculate_reward(self):
        if self.cpu_hp <= 0:
            return 100  # Win
        elif self.player_hp <= 0:
            return -100  # Loss
        else:
            return (self.player_hp - self.cpu_hp) / 100  # Ongoing game, reward based on health difference

    def _get_observation(self):
        surf = pygame.Surface((self.width, self.height))
        surf.blit(self.screen, (0, 0))
        surf = pygame.transform.scale(surf, (84, 84))
        obs = pygame.surfarray.array3d(surf)
        obs = np.mean(obs, axis=2).astype(np.uint8)
        return obs.reshape(84, 84, 1)
        
    def _render_scoreboard(self):
        font = pygame.font.Font(None, 36)
        player_text = font.render(f'Player HP: {self.player_hp}', True, (0, 0, 255))
        cpu_text = font.render(f'CPU HP: {self.cpu_hp}', True, (255, 0, 0))
        
        # Adjust scoreboard positions to be at the very top
        self.screen.blit(player_text, (10, 10))  # Player HP at the top-left
        self.screen.blit(cpu_text, (self.width - 200, 10))  # CPU HP at the top-right