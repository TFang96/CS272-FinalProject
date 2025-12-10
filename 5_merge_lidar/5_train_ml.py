import gymnasium as gym
import torch
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import os

OUTDIR = "merge_lidar"
os.makedirs(OUTDIR, exist_ok=True)

def make_env():
    env = gym.make(
        'merge-v0', 
        render_mode=None, 
        config={
            "observation": {
                "type": "LidarObservation",
                "cells": 32,
                "maximum_range": 64,
                "normalise": True
            }
        }
    )
    return env

def main():
    env = make_env()
    env = Monitor(env, f"{OUTDIR}/monitor.csv")

    model = DQN(
        policy="MlpPolicy",
        env=env,
        target_update_interval=750,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=100000,
        device="auto",
        verbose=1
    )

    model.learn(total_timesteps=60_000)
    model.save(f"{OUTDIR}/model.zip")

    print("Training complete! Files saved to:", OUTDIR)

if __name__ == "__main__":
    main()