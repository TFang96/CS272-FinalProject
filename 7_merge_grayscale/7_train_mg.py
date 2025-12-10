import gymnasium as gym
import torch
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

OUTDIR = "merge_grayscale"
os.makedirs(OUTDIR, exist_ok=True)

def make_env():
    env = gym.make(
        'merge-v0', 
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

    '''
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
    )
    '''

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.9999,
        gae_lambda=0.95,
        clip_range=0.3,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        verbose=1,
    )

    model.learn(total_timesteps=70_000)
    model.save(f"{OUTDIR}/model.zip")

    print("Training complete! Files saved to:", OUTDIR)

if __name__ == "__main__":
    main()