import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load monitor.csv (same folder your training puts it in)
df = pd.read_csv("intersection_lidar/monitor.csv", comment="#", header=None)
df.columns = ["reward", "length", "time"]

df["reward"] = pd.to_numeric(df["reward"], errors="coerce")

rewards = df["reward"].values

# Smooth over a window
window = 200
smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")

plt.figure(figsize=(10,5))
plt.plot(smoothed)
plt.title("Smoothed Learning Curve – Intersection (Lidar)")
plt.xlabel("Episode")
plt.ylabel(f"Mean Reward (avg over {window} episodes)")
plt.grid(True)
plt.show()
plt.savefig("intersection_lidar/smoothed_learning_curve.png")
plt.close()
