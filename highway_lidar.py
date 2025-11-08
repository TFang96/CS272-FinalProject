import gymnasium
import torch
import highway_env
from matplotlib import pyplot as plt

from stable_baselines3 import DQN

env = gymnasium.make(
    'highway-v0', 
    render_mode='rgb_array', 
    config={
        "observation": {
            "type": "LidarObservation",
        }
    }
)
#env.reset()
#plt.imshow(env.render())
#plt.show()

#model = DQN(policy='MlpPolicy', env=env, verbose=1)
#model.learn(total_timesteps=1_000)
#model.save("dqn_highway_lidar")