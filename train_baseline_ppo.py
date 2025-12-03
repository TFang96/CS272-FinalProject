import register_envs
import gymnasium as gym
import highway_env
import sys
import os
import torch as th 
from typing import Callable 
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn 

MODEL_NAME = "ppo_custom_env" 
OUTDIR = "custom_env_training"
FILE_NAME_ZIP = f"{MODEL_NAME}_final.zip"
VEC_NORM_STATS_FILE = f"{OUTDIR}/vec_normalize_stats.pkl"

PHASE1_TIMESTEPS = 100000 
PHASE2_TIMESTEPS = 400000 
PHASE3_TIMESTEPS = 150000 
TOTAL_TIMESTEPS = PHASE1_TIMESTEPS + PHASE2_TIMESTEPS + PHASE3_TIMESTEPS # 750k total

LR_PHASE1 = 5e-4 # Highest LR for initial exploration
LR_PHASE2 = 3e-4 # Medium LR for core learning (3e-4 is the SB3 default)
LR_PHASE3 = 1e-4 # Lowest LR for stable fine-tuning

# Ensure directory exists
os.makedirs(OUTDIR, exist_ok=True)

def three_phase_schedule(p1_steps: int, p2_steps: int, lr1: float, lr2: float, lr3: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:

        progress_completed = 1.0 - progress_remaining
        
        timesteps_completed = progress_completed * TOTAL_TIMESTEPS
        
        # Define the cumulative boundaries
        boundary_1 = p1_steps
        boundary_2 = p1_steps + p2_steps
        
        if timesteps_completed < boundary_1:
            return lr1
        elif timesteps_completed < boundary_2:
            return lr2
        else:
            return lr3
            
    return func

def create_env(monitor_path=None):
    """Creates and configures the custom roundabout environment."""
    env = gym.make(
        "custom-roundabout-v0",
        render_mode="rgb_array",
    )
    if monitor_path:
        env = Monitor(env, monitor_path) 
    return env

PPO_HYPERPARAMS = dict(
    learning_rate=three_phase_schedule(
        PHASE1_TIMESTEPS, 
        PHASE2_TIMESTEPS, 
        LR_PHASE1, 
        LR_PHASE2, 
        LR_PHASE3
    ), 
    n_steps=2048,
    gamma=0.999,
    batch_size=512,
)

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Setup Environment
    train_env = DummyVecEnv([lambda: create_env(monitor_path=f"{OUTDIR}/monitor.csv")])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Setup Model
    model = PPO(
        "MlpPolicy", 
        train_env,
        verbose=1,
        device="auto",
        **PPO_HYPERPARAMS
    )
    print("Starting training...")
    print(f"Phase 1 (LR={LR_PHASE1}) for {PHASE1_TIMESTEPS} timesteps.")
    print(f"Phase 2 (LR={LR_PHASE2}) for {PHASE2_TIMESTEPS} timesteps.")
    print(f"Phase 3 (LR={LR_PHASE3}) for {PHASE3_TIMESTEPS} timesteps.")
    print(f"Total training duration: {TOTAL_TIMESTEPS} timesteps.")


    # 3. Training Loop with Interruption Handling
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=1,
        )
        print("\nTraining completed successfully without interruption.")

    except KeyboardInterrupt:
        print("\n\n--- Training interrupted by user (Ctrl+C). Saving model and stats... ---")

    except Exception as e:
        print(f"\n\n--- An unexpected error occurred: {e}. Saving model and stats... ---")
    
    # 4. Save Logic
    model_save_path = f"{OUTDIR}/{FILE_NAME_ZIP}"
    stats_save_path = VEC_NORM_STATS_FILE

    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    
    print(f"Saving VecNormalize stats to {stats_save_path}...")
    train_env.save(stats_save_path) 

    print(f"\nTraining finished after {model.num_timesteps} timesteps.")
    print(f"Final model saved as {FILE_NAME_ZIP}.")

    # 5. Close the environment
    train_env.close()