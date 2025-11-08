import gymnasium
import torch
import highway_env
from matplotlib import pyplot as plt

from stable_baselines3 import DQN

env = gymnasium.make(
    'merge-v0', 
    render_mode='rgb_array', 
    config={
        "observation": {
            "type": "GrayscaleObservation", 
            "observation_shape": (84, 84), 
            "stack_size": 4, 
            "weights": [0.2989, 0.5870, 0.1140]
        }
    }
)
#env.reset()
#plt.imshow(env.render())
#plt.show()

#model = DQN(policy='MlpPolicy', env=env, verbose=1)
#model.learn(total_timesteps=1_000)
#model.save("dqn_merge_grayscale")