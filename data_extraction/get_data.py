import requests
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def save_json_from_url(url, save_dir="data/raw"):
    """
    Download JSON from a temporary Google Storage link
    and save it locally with a timestamped filename.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Timestamped filename for version control
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(save_dir) / f"game_data_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved data to {filename}")
    return filename, data

def json_to_dataframe(data):
    """
    Convert nested JSON into a flat DataFrame.
    Handles dicts/lists gracefully.
    """
    df = pd.json_normalize(data)
    df.columns = df.columns.str.replace('.', '_')
    return df.T

# Example usage
filename, data = save_json_from_url(url)   # download + save
df = json_to_dataframe(data)               # convert to DataFrame
print(df.head())
