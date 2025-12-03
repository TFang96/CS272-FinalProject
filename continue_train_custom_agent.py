import register_envs
import gymnasium as gym
import highway_env
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- Configuration ---
OUTDIR = "custom_env_training"
MODEL_NAME = "ppo_baseline_custom_env_final"
VEC_NORM_STATS_FILE = f"{OUTDIR}/vec_normalize_stats.pkl" 
monitor_log_path = f"{OUTDIR}/monitor.csv"
modelFile = f"{OUTDIR}/{MODEL_NAME}"
saveAs = f"{OUTDIR}/{MODEL_NAME}"
ADDITIONAL_TIMESTEPS = 1000000

# --- Environment Setup Function ---
def create_env():
    """Creates and configures the custom roundabout environment."""
    env = gym.make(
        "custom-roundabout-v0",
        render_mode="rgb_array",
    )
    # Use override_existing=False to append data during continued training
    env = Monitor(
        env,
        filename=monitor_log_path,
        allow_early_resets=True,
        override_existing=False, 
    )
    return env

# --- Main Execution ---
if __name__ == "__main__":
    env = DummyVecEnv([lambda: create_env()]) 

    # 1. Load VecNormalize stats
    if os.path.exists(VEC_NORM_STATS_FILE):
        print(f"Loading VecNormalize stats from {VEC_NORM_STATS_FILE}")
        env = VecNormalize.load(VEC_NORM_STATS_FILE, env)
        env.norm_obs = True
        env.norm_reward = True
        env.clip_obs = 10.
    else:
        print(f"CRITICAL ERROR: VecNormalize stats file not found at {VEC_NORM_STATS_FILE}. Exiting.")
        sys.exit(1)

    # 2. Load PPO Model
    print(f"Loading existing PPO model from {modelFile}...")
    try:
        model = PPO.load(modelFile, env=env, device="auto", custom_objects=None)
        print("Model loaded successfully. Resume training.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model could not be loaded")
        env.close() 
        sys.exit(1)

    # 3. Training Loop with Interruption Handling
    print(f"\nContinuing training for {ADDITIONAL_TIMESTEPS} more timesteps...")
    
    try:
        model.learn(
            total_timesteps=ADDITIONAL_TIMESTEPS, 
            log_interval=1,
            reset_num_timesteps=False 
        )
        # Normal completion message
        print("\nTraining completed successfully without interruption.")

    except KeyboardInterrupt:
        # Interruption handling
        print("\n\n--- Training interrupted by user (Ctrl+C). Saving model and stats... ---")

    except Exception as e:
        # Handle other unexpected errors
        print(f"\n\n--- An unexpected error occurred: {e}. Saving model and stats... ---")
    
    # 4. Save Logic (Executed on normal completion OR interrupt)
    print(f"Saving model to {saveAs}...")
    model.save(saveAs)
    
    print(f"Saving VecNormalize stats to {VEC_NORM_STATS_FILE}...")
    env.save(VEC_NORM_STATS_FILE) 

    final_timesteps = model.num_timesteps
    print(f"\nTotal cumulative timesteps trained: {final_timesteps}")
    print("Cleanup finished.")

    # 5. Close the environment
    env.close()