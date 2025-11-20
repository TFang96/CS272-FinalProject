import register_envs
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import time
import highway_env

env = gym.make(
    "roundabout-v0",
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

SIM_FREQ = env.unwrapped.config["simulation_frequency"]
PAUSE_TIME = 1 / SIM_FREQ 


def visualize_agent_performance_on_input(model, env, num_episodes=3):
    """Runs and displays multiple episodes of the trained agent, waiting for user input between episodes."""

    plt.ion() 
    
    for episode in range(num_episodes):
        
        if episode > 0:
            input("Press Enter to start the next episode...") 
            
        print(f"\n--- Running Episode {episode + 1}/{num_episodes} ---")
        
        obs, info = env.reset()
        
        fig, ax = plt.subplots()
        im = ax.imshow(env.render())
        ax.set_title(f"Episode {episode + 1}")
        plt.show()

        done = False
        step_count = 0
        total_reward = 0
        
        while not done:
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            im.set_data(env.render())
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            time.sleep(PAUSE_TIME)
            
        print(f"Episode finished after {step_count} steps. Total Reward: {total_reward:.2f}")


        time.sleep(1) 
        plt.close(fig) 

    plt.ioff() 

model = None
visualize_agent_performance_on_input(model, env, num_episodes=20)
env.close()