from stable_baselines3 import PPO
import gymnasium as gym
import highway_env
import register_envs 
import os
import torch as th 
from typing import Callable 

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn 

MODEL_NAME = "ppo_custom_env" 
OUTDIR = "custom_env_training"
FILE_NAME_ZIP = f"{MODEL_NAME}.zip"

# Ensure directory exists
os.makedirs(OUTDIR, exist_ok=True)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def create_env(monitor_path=None):
    env = gym.make(
        "custom-roundabout-v0",
        render_mode="rgb_array",
    )
    if monitor_path:
        env = Monitor(env, monitor_path) 
    return env


train_env = DummyVecEnv([lambda: create_env(monitor_path=f"{OUTDIR}/monitor.csv")])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)


policy_kwargs = dict(
    activation_fn=nn.Tanh,
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

PPO_HYPERPARAMS = dict(
    learning_rate=linear_schedule(3e-4), 
    n_steps=8196,                     
    batch_size=256, #Higher batch size to keep stable                      
    gamma=0.9999,                        
    gae_lambda=0.95,
    n_epochs=10,                        
    ent_coef=0.02,                    
    vf_coef=0.5,                        
)

model = PPO(
    "MlpPolicy", 
    train_env,
    policy_kwargs=policy_kwargs, 
    verbose=1,
    device="auto",
    **PPO_HYPERPARAMS
)
print("Starting PPO training...")

TOTAL_TIMESTEPS = 100000 
print(f"Training PPO for {TOTAL_TIMESTEPS} total timesteps.")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    log_interval=1,
)


model.save(f"{OUTDIR}/{MODEL_NAME}_final.zip")
train_env.save(f"{OUTDIR}/vec_normalize_stats.pkl") 

print(f"\n✅ Training finished after {model.num_timesteps} timesteps.")
print(f"Final model saved as {MODEL_NAME}_final.zip.")

train_env.close()
