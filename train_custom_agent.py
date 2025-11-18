from stable_baselines3 import PPO 

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,  
    device="cuda",
    n_steps=2048,         
    batch_size=64,         
    n_epochs=10             
)

print("Starting PPO training...")
model.learn(total_timesteps=200_000) 

model.save("ppo_roundabout_model.zip")
print("Model saved successfully.")