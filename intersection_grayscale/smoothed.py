import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("intersection_grayscale/monitor.csv", comment="#", header=None)
df.columns = ["reward", "length", "time"]

df["reward"] = pd.to_numeric(df["reward"], errors="coerce")

rewards = df["reward"].values

window = 200
smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")

plt.figure(figsize=(10,5))
plt.plot(smoothed)
plt.title("Smoothed Learning Curve – Intersection (Grayscale)")
plt.xlabel("Episode")
plt.ylabel(f"Mean Reward (avg over {window} episodes)")
plt.grid(True)
plt.savefig("intersection_lidar/smoothed_learning_curve.png")
plt.show()
plt.close()