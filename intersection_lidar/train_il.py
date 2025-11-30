import gymnasium as gym
import torch
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

OUTDIR = "intersection_lidar"
os.makedirs(OUTDIR, exist_ok=True)

def make_env():
    env = gym.make(
        'intersection-v1', 
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
    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 128],  # policy network (actor)
            vf=[128, 128],  # value function (critic)
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024, #shorter rollout
        batch_size=64,
        ent_coef=0.01, #this encourages more random exploration early on
        learning_rate=3e-4, # if too high, we would be trusting new info too quick
        gamma=0.99, #close to 1 -- driving needs to consider the future
        gae_lambda=0.95, # agent doesn't get confused with moment/moment rewards
        clip_range=0.2, #prevents policy from changing too much
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    # Stage 1: easy intersection
    env.unwrapped.config.update({
        "vehicles_count": 1,
        # maybe no pedestrians here if they exist
    })
    model.learn(total_timesteps=20_000, reset_num_timesteps=True)

    # Stage 2: harder intersection
    env.unwrapped.config.update({
        "vehicles_count": 5,
    })
    model.learn(total_timesteps=30_000, reset_num_timesteps=False)
    model.save(f"{OUTDIR}/model.zip")

    print("Training complete! Files saved to:", OUTDIR)

if __name__ == "__main__":
    main()