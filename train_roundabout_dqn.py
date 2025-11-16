import gymnasium as gym
import highway_env          # required for base highway-env functionality
import register_envs        # this runs the register() above and adds your env to gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

env = gym.make("roundabout-yield-exit-v0")
env.unwrapped.config.update({
    "observation": {"type": "LidarObservation"},
    "action": {"type": "DiscreteMetaAction"},
    "vehicles_count": 15,
    "target_exit": 2,
})
env.reset()
env = Monitor(env)

model = DQN(
    "MlpPolicy", env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=15000,
    learning_starts=2000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=500,
    verbose=1,
)
model.learn(total_timesteps=200_000)
model.save("models/dqn_roundabout_lidar")
