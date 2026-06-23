from private import url
from data_extraction.get_data import *
from data_extraction.process_data import *
from analysis.stats import *
from analysis.bot_analysis import *
from prediction_pipeline import *
# Example usage
filename, data = save_json_from_url(url)   # download + save
df = json_to_dataframe(data)  # convert to DataFrame
unpack_transposed_df_to_csv(df)
stats = load_stats(filename)
modes = stats['ratingType'].unique()
for mode in modes:
    results = compute_stats(stats, mode)
    print(f"{mode} stats:", results)
    save_plot(results['data'], mode, save_dir="plots")
suspect_bots = detect_bots(
    bot_file="data/statistics_stats_last10BotGames.csv",
    global_file="data/gameHistory_games.csv", save_dir="plots")
df = load_game_data("data/gameHistory_games.csv")
X, y, preprocess = preprocess_data(df)
model, acc = train_model(X, y, preprocess)
print("Model accuracy:", acc)

