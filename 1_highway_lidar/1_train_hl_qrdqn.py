import gymnasium as gym
import torch
import highway_env
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
import os

OUTDIR = "highway_lidar_qrdqn"
os.makedirs(OUTDIR, exist_ok=True)

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

def main():
    env = make_env()
    env = Monitor(env, f"{OUTDIR}/monitor.csv")
    if torch.cuda.is_available():
        device="cuda"
    else:
        print("No compatible GPU...")
        device="cpu"
    model = QRDQN(
        policy="MlpPolicy",
        env=env,
        target_update_interval=750, #reduce value overestimation
        learning_rate=3e-4, # balances learning rate and stability
        gamma=0.99, # driving requires planning -- closer to 1 to consider future rewards
        buffer_size=100000, #large enough to have a diverse set of transitions
        device=device,
        verbose=1
    )

    model.learn(total_timesteps=80_000)
    model.save(f"{OUTDIR}/model.zip")

    print("Training complete! Files saved to:", OUTDIR)

if __name__ == "__main__":
    main()