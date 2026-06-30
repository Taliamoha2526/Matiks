import ast
import pandas as pd
import json
import matplotlib.pyplot as plt

class DataValidationError(Exception):
    """Raised when input data fails validation checks."""

def load_activity_data(filepath: str) -> pd.DataFrame:
    """Load and validate activities dataset."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise DataValidationError(f"File not found: {filepath}")
    required_cols = ["date", "dateString", "totalDuration", "activityBreakdown", "statikCoinsEarned"]
    for col in required_cols:
        if col not in df.columns:
            raise DataValidationError(f"Missing required column: {col}")
    return df

def flatten_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Expand activityBreakdown into long-form DataFrame."""
    records = []
    for _, row in df.iterrows():
        try:
            line = row['activityBreakdown']
            if isinstance(line, str):
                line = ast.literal_eval(line)
            for act in line:
                records.append({
                    "date": row["date"],
                    "dateString": row["dateString"],
                    "totalDuration": int(row["totalDuration"]),
                    "statikCoinsEarned": int(row["statikCoinsEarned"]),
                    "activityName": act.get("activityType", "Unknown"),
                    "activityCoins": int(act.get("coinsEarned", 0)) })
        except Exception as e:
            print(f"Error parsing row {row['date']} : {e}")
    return pd.DataFrame(records)

def compute_activity_stats(activities: pd.DataFrame) -> dict:
    """Compute per-activity and overall statistics."""
    if "dateString" not in activities.columns:
        raise DataValidationError("dateString column missing from the dataset.")
    daily_totals = activities.groupby("dateString")['totalDuration'].sum()
    daily_coins = activities.groupby("dateString")['statikCoinsEarned'].sum()
    stats = {
        "total_days": int(activities["dateString"].nunique()),
        "total_duration": int(activities["totalDuration"].sum()),
        "total_coins": int(activities["statikCoinsEarned"].sum()),
        "avg_duration_per_day": float(daily_totals.mean()),
        "avg_coins_per_day": float(daily_coins.mean())}
    return stats

def save_activity_stats(stats: dict, filepath: str):
    """Save stats as JSON with Python-native types."""
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats → {filepath}")

def plot_activity_coins(activities: pd.DataFrame, outpath: str):
    """Plot bar chart of coins earned per activity."""
    agg = activities.groupby("activityName")["activityCoins"].sum().sort_values()
    agg.plot(kind="barh", figsize=(8,6))
    plt.title("Total Coins per Activity")
    plt.xlabel("Coins")
    plt.ylabel("Activity Type")
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved plot → {outpath}")
def plot_total_duration(df: pd.DataFrame, outpath: str):
    """Plot bar chart of total duration per day."""
    if df.empty:
        raise DataValidationError("No duration data available for plotting.")
    agg = df.copy()
    plt.figure(figsize=(12,5))
    agg_sorted = agg.sort_values(by="date")
    agg_sorted['date'] = pd.to_datetime(agg['date'])
    agg_sorted['duration_hours'] = agg_sorted['totalDuration'] / 3600000
    plt.plot(agg_sorted['date'], agg_sorted['duration_hours'])
    plt.title("Total Duration per Day")
    plt.xlabel("Date")
    plt.ylabel("Duration(hours)")
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved plot → {outpath}")