import gymnasium as gym
import torch
import highway_env
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

OUTDIR = "highway_grayscale"
os.makedirs(OUTDIR, exist_ok=True)

class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[2] if len(observation_space.shape) == 3 else 1

        def _conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.conv_net = nn.Sequential(
            _conv_block(n_channels, 16),
            _conv_block(16, 32),
            _conv_block(32, 32),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, observation_space.shape[0], observation_space.shape[1])
            conv_out = self.conv_net(dummy)
            conv_out_flat = conv_out.view(1, -1)
            conv_out_size = conv_out_flat.shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.float() / 255.0 if observations.dtype == torch.uint8 else observations.float()

        if obs.ndim == 4:
            obs = obs.permute(0, 3, 1, 2)  

        x = self.conv_net(obs)
        return self.fc(x)

def make_env():
    env = gym.make(
        'highway-v0', 
        render_mode=None, 
        config={
            "observation": {
                "type": "GrayscaleObservation", 
                "observation_shape": (84, 84), 
                "stack_size": 4, 
                "weights": [0.2989, 0.5870, 0.1140]
            }
        }
    )
    return env

def main():
    env = make_env()
    env = Monitor(env, f"{OUTDIR}/monitor.csv")

    model = QRDQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=30000,
        learning_starts=5000,
        batch_size=32,
        gamma=0.98,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        verbose=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_steps=1,
        policy_kwargs=dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(features_dim=256),
        ),
    )

    model.learn(total_timesteps=70_000)
    model.save(f"{OUTDIR}/model.zip")

    print("Training complete! Files saved to:", OUTDIR)

if __name__ == "__main__":
    main()