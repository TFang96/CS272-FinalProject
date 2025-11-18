import register_envs
import gymnasium
import highway_env
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
    
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-3, # Use a reasonable learning rate
    )

    print("Training DQN model...")
    model.learn(total_timesteps=total_timesteps)

    print("Evaluating trained policy...")
    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(), # Use the vectorized environment from the model
        n_eval_episodes=10,
        render=False
    )

    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Dev Reward: {std_reward:.2f}")

    if mean_reward > 0.0: # Check if the agent performs significantly better than a random agent (which might get around 0 or slightly negative)
        print("Learnable")
    else:
        print("Unlearnable")

if __name__ == "__main__":
    
    print(f"Starting test for custom environment")
    
    env = None
    try:
        env = create_env()

        test_api_compliance(env)

        test_learnability(env, total_timesteps=10000)

    except Exception as e:
        print(f"\ERROR: One or more tests failed: {e}")
        
    finally:
        if env is not None:
            env.close()
