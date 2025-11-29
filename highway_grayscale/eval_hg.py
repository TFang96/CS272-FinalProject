import gymnasium as gym
import highway_env
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN
from sb3_contrib import QRDQN

OUTDIR = "highway_grayscale"
MODEL_PATH = f"{OUTDIR}/model.zip"
N_EPISODES = 500

def make_env():
    env = gym.make(
        'highway-v0', 
        render_mode=None, 
        config={
            "observation": {
                "type": "GrayscaleObservation", 
                "observation_shape": (84, 84), 
                "stack_size": 4, 
                "weights": [0.2989, 0.5870, 0.1140]
            }
        }
    )
    return env

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QRDQN.load(MODEL_PATH, device=device)
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
    plt.title("Highway (Grayscale) – 500 Episode Evaluation")
    plt.ylabel("Episode Return")
    plt.savefig(f"{OUTDIR}/violin_plot.png")
    plt.close()

if __name__ == "__main__":
    results = evaluate()
    np.save(f"{OUTDIR}/returns.npy", results)
    plot_violin(results)

    print("Evaluation complete! Saved:")
    print("returns.npy, violin_plot.png")
