import register_envs
import gymnasium
import highway_env
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


def create_env():
    env = gymnasium.make(
    'custom-roundabout-v0',
    render_mode='rgb_array',
    config={
    "observation": {
        "type": "TimeToCollision"
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "incoming_vehicle_destination": None,
    "duration": 11, # [s] If the environment runs for 11 seconds and still hasn't done(vehicle is crashed), it will be truncated. "Second" is expressed as the variable "time", equal to "the number of calls to the step method" / policy_frequency.
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px] width of the pygame window
    "screen_height": 1000,  # [px] height of the pygame window
    "centering_position": [0.5, 0.6],  # The smaller the value, the more southeast the displayed area is. K key and M key can change centering_position[0].
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

def test_learnability(env, total_timesteps=10000):
    print(f"\n--- Starting Learnability Test ({total_timesteps} timesteps) ---")
    
    random_mean_reward, random_std_reward = evaluate_random_policy(env, n_eval_episodes=10)
    print(f"Random Policy Mean Reward: {random_mean_reward:.2f} +/- {random_std_reward:.2f}")

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3, # Use a reasonable learning rate
        device="cuda"
    )

    print("Training DQN model...")
    model.learn(total_timesteps=total_timesteps)

    print("\nEvaluating Trained DQN Policy...")
    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(), 
        n_eval_episodes=10,
        render=False
    )

    print(f"\nEvaluation Results:")
    print(f"Trained Policy Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Comparison for 'Learnable' status
    if mean_reward > random_mean_reward: 
        print("Learnable: Trained agent performs better than random.")
    else:
        print("Unlearnable: Trained agent does not significantly outperform random.")

def evaluate_random_policy(env, n_eval_episodes=10, render=False):
    """
    Evaluates a random agent over n_eval_episodes.
    Returns the mean reward and standard deviation.
    """
    print("Evaluating Random Policy...")
    all_episode_rewards = []
    
    # Run evaluation for the specified number of episodes
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 1. Choose a random action
            action = env.action_space.sample()
            
            # 2. Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # 3. Optional rendering
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
