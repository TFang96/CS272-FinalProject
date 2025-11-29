import gymnasium as gym
import torch
import highway_env
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
import os

OUTDIR = "highway_grayscale"
os.makedirs(OUTDIR, exist_ok=True)

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

def main():
    env = make_env()
    env = Monitor(env, f"{OUTDIR}/monitor.csv")

    '''
    policy_kwargs = dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=256),
        dueling=True,
        noisy=True
    )
    '''

    '''
    model = DQN(
        "CnnPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=2000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    '''

    model = QRDQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=30000,
        learning_starts=5000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    model.learn(total_timesteps=50_000)
    model.save(f"{OUTDIR}/model.zip")

    print("Training complete! Files saved to:", OUTDIR)

if __name__ == "__main__":
    main()