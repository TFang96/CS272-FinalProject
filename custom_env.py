import gymnasium
import torch
import highway_env
from matplotlib import pyplot as plt

from stable_baselines3 import DQN

env = gymnasium.make(
    'customenv-v1',
    render_mode='rgb_array',
    config={
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84),
            "stack_size": 4,
            "weights": [0.2989, 0.587, 0.1140]
        }
    }
)

env.unwrapped.config.update({
    "collision": -5,
    "approach_round_fast": -0.5,
    "unsafe_distance": -0.3,
    "terminal": 3,
    "step": -0.01
})