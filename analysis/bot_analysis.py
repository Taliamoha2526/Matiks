import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a CSV file safely."""
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"Dataset not found at {file}")
    return pd.read_csv(file)


def detect_bots(bot_file: str, global_file: str, save_dir: str = "analysis/graphs") -> pd.DataFrame:
    """
    Detect suspect bots based on similarity scores and plot results.
    Raises an error if bot dataset is missing.
    """
    # Load datasets
    try:
        bot_data = load_dataset(bot_file)
    except FileNotFoundError:
        raise ValueError("Bot dataset is required but not found. Please provide a valid bot file.")

    global_data_all = load_dataset(global_file)
    global_data = global_data_all.drop_duplicates(subset="opponentId", keep="last")

    # Extract IDs
    bot_ids = bot_data["opponentId"].astype(str).unique()
    all_ids = global_data["opponentId"].astype(str).unique()

    # Vectorize IDs
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    vectorizer.fit(all_ids)

    bot_vectors = vectorizer.transform(bot_ids).toarray()
    if bot_vectors.size == 0:
        raise ValueError("Bot dataset has no valid opponent IDs.")

    bot_fingerprint = np.mean(bot_vectors, axis=0).reshape(1, -1)
    all_vectors = vectorizer.transform(all_ids).toarray()

    # Similarity scores
    global_data["similarity_score"] = cosine_similarity(all_vectors, bot_fingerprint).flatten()

    # Threshold detection
    mean = np.mean(global_data["similarity_score"])
    std = np.std(global_data["similarity_score"])
    threshold = mean + std

    suspect_bots = global_data[global_data["similarity_score"] >= threshold]

    # Retrieve all games of suspect bots
    ids = suspect_bots["opponentId"]
    bot_games = global_data_all[global_data_all["opponentId"].isin(ids)]

    # Plot results
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.bar(x=["Wins", "Losses"], height=bot_games["isWinner"].value_counts())
    ax.set_title("Games against bots")
    ax.set_xlabel("Result")
    filename = Path(save_dir) / "bot_analysis.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {filename}")

    return suspect_bots



