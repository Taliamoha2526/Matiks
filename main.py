from private import *
from data_extraction.get_data import *
from data_extraction.process_data import *
from analysis.stats import *
from analysis.bot_analysis import *

# Example usage
filename, data = save_json_from_url(url)   # download + save
df = json_to_dataframe(data)  # convert to DataFrame
unpack_transposed_df_to_csv(df)
stats = load_stats(filename)
modes = stats['ratingType'].unique()
for mode in modes:
    results = compute_stats(stats, mode)
    print(f"{mode} stats:", results)
    save_plot(results['data'], mode)

