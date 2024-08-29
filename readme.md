# RL SharpShooter: AI-Powered 2D Space Defender

## Custom PPO Implementation for 2D Shooter Game

PPO-SharpShooter is a reinforcement learning project that implements a Proximal Policy Optimization (PPO) agent from scratch to master a custom-built 2D shooter game. This project showcases the power of reinforcement learning in game environments and demonstrates a deep understanding of both PPO algorithm implementation and game development.

## Project Overview

In PPO-SharpShooter, a custom-built PPO agent learns to play a 2D shooter game, competing against a CPU-controlled opponent. The game environment (`ShooterEnv`) is developed using Pygame and integrated with OpenAI's Gym framework, while the PPO algorithm is implemented from the ground up using PyTorch. This project serves as a comprehensive example of applying advanced reinforcement learning techniques to game AI.

## Table of Contents

1. [Key Features](#key-features)
2. [Technologies Used](#technologies-used)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Environment: ShooterEnv](#environment-shooterenv)
6. [Custom PPO Implementation](#custom-ppo-implementation)
7. [Training the Agent](#training-the-agent)
8. [Results and Visualization](#results-and-visualization)
9. [Usage](#usage)
10. [Customization](#customization)
11. [Future Improvements](#future-improvements)
12. [Contributing](#contributing)
13. [License](#license)
14. [References](#references)

## Key Features

- **Custom PPO Implementation:** Designed from scratch, showcasing a deep understanding of the algorithm.
- **Actor-Critic Network Architecture:** Built using PyTorch to predict actions and value functions.
- **2D Shooter Game Environment:** Developed with Pygame and integrated with OpenAI Gym for easy interaction.
- **Adjustable Game Speed and Frame Skipping:** Facilitates efficient training by controlling the pace of the game.
- **Visualization of Training Progress:** Includes loss curves, episode rewards, and game play performance.
- **Detailed Documentation:** Provides both practical implementation guides and theoretical explanations.

## Technologies Used

- **Python 3.x**
- **PyTorch**
- **Pygame**
- **OpenAI Gym**
- **Matplotlib**
- **NumPy**

## Project Structure

```plaintext
├── shooter_env.py         # ShooterEnv game environment
├── ppo_agent.py           # Custom PPO implementation and Actor-Critic network
├── train.py               # Script for training the PPO agent
├── evaluate.py            # Script for evaluating a trained agent
├── models/                # Directory for saved models
├── results/               # Directory for training visualizations
├── requirements.txt       # List of required Python packages
└── README.md              # Project documentation
├── shooter_game.py        # Script to run a sample game session

```

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/jeffasante/RL-PPO-Shooter.git
   cd RL-PPO-Shooter
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Environment: ShooterEnv

### Overview

`ShooterEnv` is a custom game environment that simulates a 2D shooter game where an agent competes against a CPU-controlled opponent. It provides a standard Gym interface for reinforcement learning, making it easy to integrate with various RL algorithms.

### Key Features

- **State Space:** An 84x84 grayscale image of the game screen.
- **Action Space:** `Discrete(3)` — [Move Left, Move Right, Shoot].
- **Reward System:**
  - +100 for defeating the CPU.
  - -100 for losing to the CPU.
  - Ongoing rewards based on the health difference between the player and CPU.
- **Customizable Parameters:**
  - `speed_multiplier`: Adjusts game speed.
  - `skip_frames`: Number of frames to skip between actions.

## Custom PPO Implementation

### Overview

The PPO algorithm is implemented from scratch, showcasing a deep understanding of policy gradient methods and the specific improvements introduced by PPO.

### Key Components

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        # Shared convolutional layers
        # Separate actor and critic heads

class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, value_coef, entropy_coef):
        # Initialize PPO parameters

    def get_action(self, state):
        # Action selection logic

    def update(self, states, actions, old_log_probs, rewards, dones):
        # PPO update algorithm implementation
        # Includes clipped objective and value function loss
```

### Important Aspects

- **Advantage Estimation and Normalization:** Improves training stability.
- **PPO Clipping:** Prevents large updates, ensuring stable policy updates.
- **Separate Value Function and Entropy Loss Components:** For more efficient learning.
- **Customizable Hyperparameters:** Fine-tune the learning process to achieve better results.

## Training the Agent

The training process involves:

1. **Interacting with the `ShooterEnv`:** The agent observes the environment and takes actions.
2. **Collecting Experiences:** States, actions, rewards, and other data are collected for training.
3. **Updating the PPO Model:** The model is updated periodically using the collected experiences.
4. **Visualizing the Training Progress:** Losses and rewards are plotted to monitor the training.

```python
def train(env, ppo_agent, num_episodes, save_freq=100):
    for episode in range(num_episodes):
        # Collect episode data
        # Update PPO agent
        # Save model and plot losses periodically
```

## Results and Visualization

The training process generates two main visualizations:

1. `ppo_losses.png`: Actor loss, Critic loss, and Entropy loss over time.
2. `episode_rewards.png`: Rewards obtained in each episode during training.

These visualizations provide insights into the agent's learning progress and performance.

## Usage

### Training the Agent

```python
from shooter_env import ShooterEnv
from ppo_agent import PPO

env = ShooterEnv(speed_multiplier=2, skip_frames=2)
state_dim = (1, 84, 84)
action_dim = env.action_space.n

ppo_agent = PPO(state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01)
train(env, ppo_agent, num_episodes=1000, save_freq=100)
```

### Evaluating a Trained Agent

```python
env = ShooterEnv(speed_multiplier=1, skip_frames=1)
ppo_agent = PPO(state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01)
ppo_agent.load_model('models/ppo_model_final.pth')

state = env.reset()
done = False
while not done:
    action, _ = ppo_agent.get_action(state)
    state, reward, done, _ = env.step(action)
    env.render()
```

 **Run inference:**
   ```bash
   python evaluate.py 
   ```

## Customization

- **Modify `ShooterEnv` Parameters:** Customize the game difficulty by adjusting player and CPU health, bullet speed, etc.
- **Adjust the Actor-Critic Network Architecture:** Experiment with different network structures in the `ActorCritic` class.
- **Experiment with PPO Hyperparameters:** Fine-tune learning rates, clipping values, and other parameters for optimized training.

## Future Improvements

- **Implement Parallel Environments:** Speed up training by using multiple environments simultaneously.
- **Explore Different Network Architectures:** Test LSTMs or other architectures to capture temporal dependencies.
- **Support for Continuous Action Spaces:** Expand the environment to handle continuous action spaces.
- **Compare with Other RL Algorithms:** Implement A2C, SAC, or other algorithms to benchmark performance.
- **Enhance the CPU Opponent's AI:** Make the CPU a more formidable opponent by improving its decision-making capabilities.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you encounter any problems.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
- Pygame Documentation: [Pygame Docs](https://www.pygame.org/docs/)
- PyTorch Documentation: [PyTorch Docs](https://pytorch.org/docs/)
- OpenAI Gym: [OpenAI Gym](https://gym.openai.com/)
