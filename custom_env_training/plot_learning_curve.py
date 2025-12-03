import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MONITOR_PATH = "monitor.csv"
OUT_PLOT = "learning_curve.png"
WINDOW_SIZE = 100 # Number of episodes to average over

def main():
    df = pd.read_csv(MONITOR_PATH, skiprows=1)
    df["episode"] = range(1, len(df) + 1)

    # 1. Calculate the Moving Average
    df["rolling_mean"] = df["r"].rolling(
        window=WINDOW_SIZE, 
        min_periods=1, 
        center=True
    ).mean()

    # 2. Fit a Linear Regression Line to the Rolling Mean
    
    # Extract data points for fitting
    # We use the rolling mean data for a smoother fit
    x = df["episode"].values
    y = df["rolling_mean"].values
    
    # Remove NaN values that might occur at the very start (if min_periods > 1)
    # This step is mostly defensive, as min_periods=1 is used.
    valid_indices = ~np.isnan(y)
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    
    # Fit a 1st degree polynomial (a straight line: y = m*x + c)
    # The result (p) is an array containing [slope, intercept]
    p = np.polyfit(x_valid, y_valid, 1)
    
    # Create the polynomial function object
    poly_func = np.poly1d(p)
    
    # Calculate the y-values for the fitted line
    df["linear_trend"] = poly_func(df["episode"])


    plt.figure(figsize=(10, 5))
    
    # 3. Plot the Raw Data (Noisy)
    plt.plot(
        df["episode"], 
        df["r"], 
        label="Raw Episode Return (r)", 
        alpha=0.3, # Made more transparent to focus on the trend
        color='blue'
    )
    
    # 4. Plot the Rolling Mean (Smoothed Trend)
    plt.plot(
        df["episode"], 
        df["rolling_mean"], 
        label=f"{WINDOW_SIZE}-Episode Moving Average", 
        color='red', 
        linewidth=2
    )

    # 5. Plot the Linear Slope Line
    plt.plot(
        df["episode"], 
        df["linear_trend"], 
        label=f"Overall Linear Trend (Slope: {p[0]:.4f})", 
        color='green', 
        linestyle='--', # Dotted line
        linewidth=2
    )
    
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curve – Trend Analysis (Raw, Rolling Mean, and Linear Fit)")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_PLOT)
    plt.close()

    print("Learning curve saved to:", OUT_PLOT)

if __name__ == "__main__":
    main()