from get_data import df
import pandas as pd
import json
from pathlib import Path

def unpack_transposed_df_to_csv(df, save_dir="data"):
    """
    Iterate over a transposed DataFrame (features as rows).
    - Flat values (strings, ints, floats) go into metadata.csv
    - Nested dicts/lists are expanded into separate CSVs
    Each nested section is saved as its own CSV in the data folder.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    meta = {}

    for feature, row in df.iterrows():
        value = row.values[0]  # single value per row

        # Try to parse JSON strings into dict/list
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                value = parsed
            except Exception:
                pass

        # Save nested structures immediately
        if isinstance(value, (dict, list)):
            try:
                nested_df = pd.json_normalize(value)
            except Exception:
                nested_df = pd.DataFrame(value)

            filename = Path(save_dir) / f"{feature}.csv"
            nested_df.to_csv(filename, index=False)
            print(f"Saved nested data for '{feature}' → {filename}")
        else:
            meta[feature] = value

    # Save flat metadata
    meta_df = pd.DataFrame([meta])
    meta_file = Path(save_dir) / "metadata.csv"
    meta_df.to_csv(meta_file, index=False)
    print(f"Saved metadata → {meta_file}")

