import register_envs
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
import time
import highway_env
import numpy as np
env = gym.make("custom-roundabout-v0")
total_returns = []
for _ in range(1000):
    obs, info = env.reset()
    done = False
    r_sum = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        r_sum += reward
    total_returns.append(r_sum)
# Check the mean and min/max
print(f"Random Agent Mean Return: {np.mean(total_returns):.2f}")