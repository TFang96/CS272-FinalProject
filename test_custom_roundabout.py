import register_envs
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import highway_env
import os # Import os for path checking

modelFile = "ppo_custom_roundabout_model_2.zip" 

def create_env():
    """
    Creates and configures the custom roundabout environment.
    NOTE: The config MUST match the one used during training in train_ppo.py.
    """
    env = gym.make(
        'custom-roundabout-v0',
        render_mode='rgb_array',
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
        "duration": 15,
        "normalize_reward": False,
        }
    )
    return env

def test_api_compliance(env):
    try:
        check_env(env, warn=True, skip_render_check=True)
        print("check_env passed")
    except Exception as e:
        print(f"check_env failed: {e}")
        raise

def test_learnability(env, total_timesteps=1):
    print(f"\n--- Starting Learnability Test ---")
    n_eval_episodes = 100
    
    #Evaluate random policy
    random_mean_reward, random_std_reward, random_crashes = evaluate_random_policy(env, n_eval_episodes=n_eval_episodes)
    print(f"Random Policy Mean Reward: {random_mean_reward:.2f} +/- {random_std_reward:.2f}")

    #Evaluate trained policy
    ppo_crashes = 0
    mean_reward = 0.0
    std_reward = 0.0

    if not os.path.exists(modelFile):
        print(f"\nSkipping PPO evaluation: Model file '{modelFile}' not found.")
        print(f"Please ensure the trained model is available for comparison.")
        return
        
    try:
        # Load model
        model = PPO.load( 
            modelFile,
            env=env,
            device="cuda"
        )

        print("PPO model loaded successfully!")

        print("\nEvaluating Trained PPO Policy...")
        ppo_rewards = []
        
        for _ in range(n_eval_episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            
            while not terminated and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
            ppo_rewards.append(episode_reward)
            # Since _is_terminated is defined as vehicle crash in the custom env
            if terminated:
                ppo_crashes += 1
                
        mean_reward = np.mean(ppo_rewards)
        std_reward = np.std(ppo_rewards)


        print(f"\nEvaluation Results:")
        print(f"Trained Policy Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        if mean_reward > random_mean_reward: 
            print("Learnable: Trained agent performs better than random")
        else:
            print("Unlearnable: Trained agent does not significantly outperform random.")
            
    except Exception as e:
        print(f"Could not load or evaluate model: {e}")
    
    print(f"\n--- Crash Analysis (N={n_eval_episodes} episodes) ---")
    
    print(f"Random Policy (Crashes/Total Episodes): {random_crashes} / {n_eval_episodes}")
    
    #Only if model was loaded
    if os.path.exists(modelFile):
        print(f"PPO Policy (Crashes/Total Episodes): {ppo_crashes} / {n_eval_episodes}")
        
def evaluate_random_policy(env, n_eval_episodes=100, render=False):
    """
    Evaluates a random agent over n_eval_episodes.
    Returns the mean reward, standard deviation, and number of crashed episodes.
    """
    print("Evaluating Random Policy...")
    all_episode_rewards = []
    num_crashes = 0
    
    for episode in range(n_eval_episodes):
        # Must reset the environment after each episode
        obs, info = env.reset() 
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not terminated and not truncated:
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
        
        all_episode_rewards.append(episode_reward)
        # Assuming terminated means a crash in the custom environment
        if terminated:
             num_crashes += 1

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    
    return mean_reward, std_reward, num_crashes

if __name__ == "__main__":
    
    print(f"Starting test for custom environment")
    
    env = None
    try:
        env = create_env()

        test_api_compliance(env)

        test_learnability(env, total_timesteps=1) 

    except Exception as e:
        print(f"\nERROR: One or more tests failed: {e}")
        
    finally:
        if env is not None:
            env.close()