from stable_baselines3 import PPO
import gymnasium as gym
import highway_env
import register_envs # Ensure your custom environment is registered

def create_env():
    """Creates and configures the custom roundabout environment."""
    env = gym.make(
        "custom-roundabout-v0",
        render_mode="rgb_array",
        config={
     
            "observation": {
                "type": "Kinematics",
            },
            
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [0, 5, 10, 15, 20]
            },
            
            "collision_reward": -5,         # Harsh penalty for crashes
            "high_speed_reward": 0.3,       # Strong incentive for speed
            "progress_reward": 0.5,        
            "time_penalty": -0.05,         
            
            "duration": 11,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 1000,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False,
            "normalize_reward": False, 
        }
    )
    return env

# Create the environment instance
env = create_env()


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4, # Standard and stable PPO learning rate
    n_steps=1024,       # Collect 2048 steps before performing an update
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,

    n_epochs=20,         
    vf_coef=1.0,         
    clip_range_vf=0.2,

    device="auto",      # Use CUDA if available
)

print("Starting PPO training...")


TOTAL_TIMESTEPS = 500000 
model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)

# Save the trained model
model.save("ppo_custom_roundabout_model_1.zip")

print(f"PPO training finished after {TOTAL_TIMESTEPS} timesteps.")
print("PPO model saved successfully as ppo_custom_roundabout_model.zip.")

env.close()