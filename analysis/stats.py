import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_stats(file_path="data/statistics_ratingHistory.csv"):
    """Load rating history CSV into a DataFrame."""
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"{file} not found.")
    return pd.read_csv(file)

def compute_stats(df, mode):
    """Compute statistics for a given ratingType mode."""
    i_data = df[df['ratingType'] == mode].copy()
    return {
        'avg_change': np.mean(i_data['ratingChange']),
        'w/l_ratio': np.sum(i_data['wins']) / np.sum(i_data['losses']),
        'max_rating': np.max(i_data['rating']),
        'min_rating': np.min(i_data['rating']),
        'data': i_data
    }
def save_plot(i_data, mode, save_dir="analysis/graphs"):
    """Plot rating history for a mode and save to analysis/graphs."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    i_data['date'] = pd.to_datetime(i_data['date'])
    plt.figure(figsize=(8, 4))
    plt.plot(i_data['date'], i_data['rating'])
    plt.title(f"{mode} rating")
    plt.xlabel("Date")
    plt.ylabel("Rating")

    filename = Path(save_dir) / f"{mode}_rating.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {filename}")



