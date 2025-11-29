import gymnasium as gym
import highway_env
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN

OUTDIR = "highway_lidar"
MODEL_PATH = "model.zip"
N_EPISODES = 500

def make_env():
    env = gym.make(
        'highway-v0', 
        render_mode=None, 
        config={
            "observation": {
                "type": "LidarObservation",
                "cells": 16,
                "maximum_range": 64,
                "normalise": True
            }
        }
    )
    return env

def evaluate():
    model = DQN.load(MODEL_PATH)
    env = make_env()

    returns = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        ep_ret = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward

            if terminated or truncated:
                returns.append(ep_ret)
                break

    env.close()
    return np.array(returns)

def plot_violin(returns):
    plt.figure(figsize=(7,6))
    sns.violinplot(data=returns)
    plt.title("Highway (Lidar) – 500 Episode Evaluation")
    plt.ylabel("Episode Return")
    plt.savefig(f"{OUTDIR}/violin_plot.png")
    plt.close()

if __name__ == "__main__":
    results = evaluate()
    np.save(f"{OUTDIR}/returns.npy", results)
    plot_violin(results)

    print("Evaluation complete! Saved:")
    print("returns.npy, violin_plot.png")
