import gymnasium as gym
import torch
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

OUTDIR = "intersection_grayscale"
os.makedirs(OUTDIR, exist_ok=True)

def make_env():
    env = gym.make(
        'intersection-v1', 
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

def main():
    env = make_env()
    env = Monitor(env, f"{OUTDIR}/monitor.csv")

    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=0.0003,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    model.learn(total_timesteps=100_000)
    model.save(f"{OUTDIR}/model.zip")

    print("Training complete! Files saved to:", OUTDIR)

if __name__ == "__main__":
    main()