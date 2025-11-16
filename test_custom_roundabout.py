import register_envs
import gymnasium
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


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
if __name__ == "__main__":
    
    print(f"Starting test for custom environment")
    
    env = None
    try:
        env = create_env()

        test_api_compliance(env)


    except Exception as e:
        print(f"\ERROR: One or more tests failed: {e}")
        
    finally:
        if env is not None:
            env.close()
