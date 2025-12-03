import register_envs
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
import time
import highway_env

OUTDIR = "custom_env_training"
MODEL_PATH = OUTDIR + "/ppo_baseline_custom_env_final"

env = gym.make(
        "custom-roundabout-v0",
        render_mode="rgb_array",
    )

SIM_FREQ = env.unwrapped.config["simulation_frequency"]
PAUSE_TIME = 1 / SIM_FREQ 



model = PPO.load(
    MODEL_PATH,
    env=env,
    device="cuda",
)

print("model loaded successfully!")

def visualize_agent_performance_on_input(model, env, num_episodes=3):

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
        
        # print(f"{'Step':<5} | {'Total Reward':<12} | {'Collision (-1)':<15} | {'High Speed (0.2)':<16} | {'Progress (0.5)':<16} | {'Lane Change (-0.05)':<20} | {'Time Penalty (-0.1)':<20}")
        # print("-" * 88)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            print(f"Action chosen: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            # reward_breakdown = info.get("rewards", {})
            # print(
            #     f"{step_count:<5} Action: {action} | {reward:.4f}     | "
            #     f"{reward_breakdown.get('collision_reward', 0):<10.4f} | "
            #     f"{reward_breakdown.get('high_speed_reward', 0):<12.4f} | "
            #     f"{reward_breakdown.get('lane_change_reward', 0):<12.4f}"
            # )

            im.set_data(env.render())
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            time.sleep(PAUSE_TIME)
            
        print(f"Episode finished after {step_count} steps. Total Reward: {total_reward:.2f}")


        time.sleep(1) 
        plt.close(fig) 

    plt.ioff() 

visualize_agent_performance_on_input(model, env, num_episodes=20)
env.close()