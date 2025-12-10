from stable_baselines3 import PPO
from sb3_contrib import QRDQN
import gymnasium as gym
import highway_env
import register_envs 
import os
from stable_baselines3.common.monitor import Monitor

fileName = "qrdqn_custom_env_normalized_reward_nov30_training.zip"

OUTDIR = "custom_env_training"
os.makedirs(OUTDIR, exist_ok=True)

def create_env():
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

# Create the environment instance
env = create_env()

env = Monitor(env, f"{OUTDIR}/monitor.csv")


policy_kwargs = dict(n_quantiles=50)
model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
print("Starting training...")


TOTAL_TIMESTEPS = 100000
model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=1)

# Save the trained model
model.save(f"{OUTDIR}/{fileName}")
print(f"PPO training finished after {TOTAL_TIMESTEPS} timesteps.")
print(f"PPO model saved successfully as {fileName}")

env.close()