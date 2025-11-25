import pandas as pd
import matplotlib.pyplot as plt

MONITOR_PATH = "highway_lidar/monitor.csv"
OUT_PLOT = "highway_lidar/learning_curve.png"

def main():
    df = pd.read_csv(MONITOR_PATH, skiprows=1)
    df["episode"] = range(1, len(df) + 1)

    plt.figure(figsize=(10,5))
    plt.plot(df["episode"], df["r"])
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curve – Highway (Lidar)")
    plt.grid(True)
    plt.savefig(OUT_PLOT)
    plt.close()

    print("Learning curve saved to:", OUT_PLOT)

if __name__ == "__main__":
    main()
