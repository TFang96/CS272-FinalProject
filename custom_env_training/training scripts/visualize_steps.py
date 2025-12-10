import register_envs
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import time
import numpy as np 

modelFile = "ppo_custom_roundabout_model_2.zip"

env = gym.make(
        'custom-roundabout-v0',
        render_mode='rgb_array',
    )


try:
    model = PPO.load(
        modelFile,
        env=env,
        device="auto", 
    )
    print("model loaded successfully!")
except Exception as e:
    print(f"Error loading model from {modelFile}: {e}")
    print("Exiting script.")
    exit()

def visualize_agent_performance_on_input(model, env, num_episodes=3):

    plt.ion()
    
    for episode in range(num_episodes):
        
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")
        
        obs, info = env.reset()
        
        fig, ax = plt.subplots()
        im = ax.imshow(env.render())
        
        done = False
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0
        
        ax.set_title(f"Episode {episode + 1}, Step 0, Total Reward: {total_reward:.2f}")
        plt.show()

        while not done:
            
            prompt = f"Episode {episode + 1} | Step {step_count + 1}. Press Enter to advance (or type 'q' to quit): "
            user_input = input(prompt)
            
            if user_input.lower() == 'q':
                print("Episode terminated by user.")
                break

            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            im.set_data(env.render())
            ax.set_title(f"Episode {episode + 1} | Step {step_count} | Total Reward: {total_reward:.2f}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        
        print(f"Episode finished after {step_count} steps. Final Reward: {total_reward:.2f}")
        
        time.sleep(1) 
        plt.close(fig)

    plt.ioff() 
    print("Visualization complete.")


visualize_agent_performance_on_input(model, env, num_episodes=15) 
env.close()