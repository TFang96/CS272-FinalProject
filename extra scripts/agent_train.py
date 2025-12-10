import create_env
from stable_baselines3 import DQN


model = DQN('MlpPolicy', create_env.create_highway_env(),
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.95,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="highway_dqn/",
            device="auto")
model.learn(total_timesteps=200000)
model = DQN('MlpPolicy', create_env.create_intersection_env(),
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.95,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="highway_dqn/",
            device="auto")
model.learn(total_timesteps=200000)
model = DQN('MlpPolicy', create_env.create_merge_env(),
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.95,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="highway_dqn/",
            device="auto")
model.learn(total_timesteps=200000)


model.save("highway_dqn/model")
