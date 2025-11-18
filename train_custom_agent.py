from stable_baselines3 import DQN
import register_envs
import gymnasium as gym

env = gym.make(
    "custom-roundabout-v0",
    render_mode="rgb_array",
    config={
        "observation": {
            "type": "TimeToCollision"
        },
        "action": {
            "type": "DiscreteMetaAction"
        },
        # ... (other config parameters) ...
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
        "offscreen_rendering": False
    }
)

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    device="cuda",
    buffer_size=100000 
)

print("Starting DQN training...")
model.learn(total_timesteps=200000)

model.save("dqn_roundabout_model.zip")
model.save_replay_buffer("dqn_roundabout_buffer.pkl") 

print("DQN model and replay buffer saved successfully.")