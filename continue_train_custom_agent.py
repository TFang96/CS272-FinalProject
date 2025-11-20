from stable_baselines3 import PPO
import gymnasium as gym
import highway_env
import register_envs 
import sys

# Name of the model file to load and save
modelFile = "ppo_custom_roundabout_model_2.zip"
saveAs = "ppo_custom_roundabout_model_2.zip"
def create_env():
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
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
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
    return env

env = create_env()

# 2. Load the previously trained model
print(f"Loading existing PPO model from {modelFile}...")
try:
    # Set custom_objects to None to avoid issues if custom components were used
    model = PPO.load(modelFile, env=env, device="auto", custom_objects=None)
    print("Model loaded successfully. Resume training.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Model could not be loaded")
    env.close() # Ensure environment cleanup
    sys.exit(1)


ADDITIONAL_TIMESTEPS = 10000
print(f"\nContinuing PPO training for {ADDITIONAL_TIMESTEPS} more timesteps...")

# Use reset_num_timesteps=False to ensure the internal timestep counter continues from where it left off
model.learn(total_timesteps=ADDITIONAL_TIMESTEPS, log_interval=4, reset_num_timesteps=False)

# 4. Save the updated model
model.save(saveAs)

print(f"\nContinued training finished.")
print(f"Model saved and overwritten successfully as {saveAs}.")

# 5. Close the environment
env.close()