import register_envs
import gymnasium as gym
import matplotlib.pyplot as plt

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
#env.reset()
#plt.imshow(env.render())
#plt.show()

import time

obs, info = env.reset()

plt.ion()  # turn on interactive mode
fig, ax = plt.subplots()
im = ax.imshow(env.render())  # initial frame
plt.show()

done = False
while not done:
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    im.set_data(env.render())
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)  

env.close()