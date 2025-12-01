from stable_baselines3 import PPO # Import TRPO
import gymnasium as gym
import highway_env
import register_envs 
import os
import torch as th 
from typing import Callable 
from sb3_contrib import TRPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn 


MODEL_NAME = "trpo_custom_env" 
OUTDIR = "custom_env_training"
FILE_NAME_ZIP = f"{MODEL_NAME}.zip"

# Ensure directory exists
os.makedirs(OUTDIR, exist_ok=True)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    # TRPO often doesn't use a schedule for the learning rate, but we keep the helper
    # in case other parameters need scheduling.
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def create_env(monitor_path=None):
    """Creates and configures the custom roundabout environment."""
    env = gym.make(
        "custom-roundabout-v0",
        render_mode="rgb_array",
        config={
                "observation": {
                    "type": "Kinematics",
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },
                },
                "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 5, 10, 15, 20]},
                "incoming_vehicle_destination": None,
                "collision_reward": -1,
                "high_speed_reward": 0.2,
                "right_lane_reward": 0,
                "lane_change_reward": -0.05,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "duration": 20,
                "normalize_reward": True,
            }
    )
    if monitor_path:
        env = Monitor(env, monitor_path) 
    return env


train_env = DummyVecEnv([lambda: create_env(monitor_path=f"{OUTDIR}/monitor.csv")])
# Normalization is often more critical for on-policy methods like TRPO
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)


policy_kwargs = dict(
    activation_fn=nn.Tanh,
    net_arch=dict(pi=[256, 256], vf=[256, 256]) # Two layers of 256 neurons for Actor (pi) and Critic (vf)
)

# TRPO Hyperparameters - Adjusted from PPO
TRPO_HYPERPARAMS = dict(
    # learning_rate is not directly used by TRPO in the same way as PPO
    # TRPO uses the conjugate gradient algorithm and line search internally.
    n_steps=8196, # Keep large n_steps for stable sampling                      
    gamma=0.9999,                        
    gae_lambda=0.95,
    # TRPO Specific Hyperparameters:
    # TRPO uses a maximum KL divergence constraint (max_kl) instead of clipping
    target_kl=0.01,
    # TRPO does not use n_epochs or batch_size as it's an on-policy, single-update algorithm
)

# Instantiate TRPO instead of PPO
model = TRPO(
    "MlpPolicy", 
    train_env,
    policy_kwargs=policy_kwargs, 
    verbose=1,
    device="auto",
    **TRPO_HYPERPARAMS
)
print("Starting TRPO training...")

TOTAL_TIMESTEPS = 10000 
print(f"Training TRPO for {TOTAL_TIMESTEPS} total timesteps.")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    log_interval=1,
)


model.save(f"{OUTDIR}/{MODEL_NAME}_final.zip")
train_env.save(f"{OUTDIR}/vec_normalize_stats.pkl") 

print(f"\n✅ Training finished after {model.num_timesteps} timesteps.")
print(f"Final model saved as {MODEL_NAME}_final.zip.")

train_env.close()