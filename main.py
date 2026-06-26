
from get_data import save_json_from_url, json_to_dataframe
from process_data import unpack_transposed_df_to_csv
from stats import load_stats, compute_stats, save_plot
from bot_analysis import detect_bots
from prediction_pipeline import load_game_data, preprocess_data, train_model, save_pipeline_artifacts
from private import *
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
# Extraction
try:
    filename, data = save_json_from_url(link)
    df = json_to_dataframe(data)
    unpack_transposed_df_to_csv(df)
except Exception as e:
    print("Extraction failed:", e)

# Stats
try:
    stats = load_stats(DATA_DIR / "statistics_ratingHistory.csv")
    modes = stats['ratingType'].unique()
    for mode in modes:
        results = compute_stats(stats, mode)
        save_plot(results, mode, save_dir=PLOTS_DIR)
except Exception as e:
    print("Stats analysis failed:", e)

# Bot analysis
try:
    suspect_bots = detect_bots(
        bot_file=DATA_DIR / "statistics_stats_last10BotGames.csv",
        global_file=DATA_DIR / "gameHistory_games.csv",
        save_dir=PLOTS_DIR
    )
    print("Suspect bots detected:", suspect_bots)
except Exception as e:
    print("Bot analysis failed:", e)

# Model training
try:
    df = load_game_data(DATA_DIR / "gameHistory_games.csv")
    X, y, preprocess = preprocess_data(df)
    model, acc, history, losses = train_model(X, y, preprocess)
    print("Model accuracy:", acc)
    save_pipeline_artifacts(model, acc, history, losses)

except Exception as e:
    print("Model training failed:", e)
