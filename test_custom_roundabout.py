import register_envs
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import highway_env
modelFile = "ppo_custom_roundabout_model_1.zip" 

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
            "type": "Kinematics"
        },
        
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20] 
        },
        
        "collision_reward": -5,
        "high_speed_reward": 0.3,
        "progress_reward": 0.5,
        "time_penalty": -0.05,
        "normalize_reward": False,

        "incoming_vehicle_destination": None,
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
    
    random_mean_reward, random_std_reward = evaluate_random_policy(env, n_eval_episodes=100)
    print(f"Random Policy Mean Reward: {random_mean_reward:.2f} +/- {random_std_reward:.2f}")

    try:
        # PPO MODEL LOADING
        model = PPO.load( 
            modelFile,
            env=env,
            device="cuda"
        )

        print("PPO model loaded successfully!")

        print("\nEvaluating Trained PPO Policy...")
        mean_reward, std_reward = evaluate_policy(
            model,
            model.get_env(), 
            n_eval_episodes=100,
            render=False
        )

        print(f"\nEvaluation Results:")
        print(f"Trained Policy Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Comparison for 'Learnable' status
        if mean_reward > random_mean_reward: 
            print("Learnable: Trained agent performs better than random")
        else:
            print("Unlearnable: Trained agent does not significantly outperform random.")
            
    except Exception as e:
        print(f"Could not load or evaluate model: {e}")
        
def evaluate_random_policy(env, n_eval_episodes=100, render=False):
    """
    Evaluates a random agent over n_eval_episodes.
    Returns the mean reward and standard deviation.
    """
    print("Evaluating Random Policy...")
    all_episode_rewards = []
    
    for episode in range(n_eval_episodes):
        # Must reset the environment after each episode
        obs, info = env.reset() 
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not terminated and not truncated:
            # 1. Choose a random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
        
        all_episode_rewards.append(episode_reward)

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    
    return mean_reward, std_reward

if __name__ == "__main__":
    
    print(f"Starting test for custom environment")
    
    env = None
    try:
        env = create_env()

        test_api_compliance(env)

        test_learnability(env, total_timesteps=1) 

    except Exception as e:
        print(f"\ERROR: One or more tests failed: {e}")
        
    finally:
        if env is not None:
            env.close()