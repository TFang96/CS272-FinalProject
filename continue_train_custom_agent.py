from stable_baselines3 import PPO
import gymnasium as gym
import highway_env
import register_envs 
import sys
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


OUTDIR = "custom_env_training"
MODEL_NAME = "ppo_custom_env_normalized_reward_nov30_training_2"
VEC_NORM_STATS_FILE = f"{OUTDIR}/vec_normalize_stats.pkl" 
monitor_log_path = f"{OUTDIR}/monitor.csv"

modelFile = f"{OUTDIR}/{MODEL_NAME}_final.zip" # Load the saved model file
saveAs = f"{OUTDIR}/{MODEL_NAME}_final.zip"     

def create_env(monitor_path=None):
    """
    Creates and configures the custom roundabout environment.
    Configuration MUST match the one used during the initial training.
    """
    env = gym.make(
        "custom-roundabout-v0",
        render_mode="rgb_array",
        config={
            "observation": {
                    "type": "Kinematics",
                    "features_range": {
                        "x": [-100, 100], "y": [-100, 100], 
                        "vx": [-15, 15], "vy": [-15, 15],
                    },
                },
                "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 5, 10, 15, 20]},
                "incoming_vehicle_destination": None,
                "collision_reward": -3,
                "high_speed_reward": 0.2,
                "progress_reward": 0.1,
                "pedestrian_proximity_reward": -0.05,
                "right_lane_reward": 0,
                "lane_change_reward": -0.05,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "duration": 20, 
                "normalize_reward": False,
        }
    )

    env = Monitor(
        env,
        filename=monitor_log_path,
        allow_early_resets=True,
        override_existing=False, # Ensures appending to monitor.csv
    )
    return env


env = DummyVecEnv([lambda: create_env()]) 

if os.path.exists(VEC_NORM_STATS_FILE):
    print(f"Loading VecNormalize stats from {VEC_NORM_STATS_FILE}")
    env = VecNormalize.load(VEC_NORM_STATS_FILE, env)
    env.norm_obs = True
    env.norm_reward = True
    env.clip_obs = 10.
else:
    print(f"CRITICAL ERROR: VecNormalize stats file not found at {VEC_NORM_STATS_FILE}. Exiting.")
    sys.exit(1)


print(f"Loading existing PPO model from {modelFile}...")
try:
    model = PPO.load(modelFile, env=env, device="auto", custom_objects=None)
    print("Model loaded successfully. Resume training.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Model could not be loaded")
    env.close() 
    sys.exit(1)


ADDITIONAL_TIMESTEPS = 50000
print(f"\nContinuing training for {ADDITIONAL_TIMESTEPS} more timesteps...")

model.learn(
    total_timesteps=ADDITIONAL_TIMESTEPS, 
    log_interval=1,
    reset_num_timesteps=False 
)

model.save(saveAs)

env.save(VEC_NORM_STATS_FILE) 

final_timesteps = model.num_timesteps
print(f"\n✅ Total cumulative timesteps trained: {final_timesteps}")

print(f"\nContinued training finished.")
print(f"Model saved and overwritten successfully as {saveAs}.")
print(f"New VecNormalize stats saved to {VEC_NORM_STATS_FILE}.")


# 5. Close the environment
env.close()