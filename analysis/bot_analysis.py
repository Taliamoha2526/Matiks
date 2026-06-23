import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
bot_data = pd.read_csv('../data/statistics_stats_last10BotGames.csv')
global_data_1 = pd.read_csv('../data/gameHistory_games.csv')
global_data = global_data_1.drop_duplicates(subset='opponentId', keep='last')

bot_ids= bot_data['opponentId'].unique()
all_ids = global_data['opponentId'].unique()

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5))
vectorizer.fit(all_ids)

bot_vectors = vectorizer.transform(bot_ids).toarray()
bot_fingerprint = np.mean(bot_vectors, axis=0).reshape(1, -1)

all_vectors = vectorizer.transform(all_ids).toarray()

global_data['similarity_score'] = cosine_similarity(all_vectors, bot_fingerprint).flatten()

#using the standard deviation method to spot the perfect threshold.
mean = np.mean(global_data['similarity_score'])
std = np.std(global_data['similarity_score'])
threshold = mean + std

suspect_bots = global_data[global_data['similarity_score'] >= threshold]
#retrieve all games of suspect bots.
ids = suspect_bots['opponentId']
bot_games = global_data_1[global_data_1['opponentId'].isin(ids)]

fig, ax = plt.subplots()
plt.bar(x = ['Wins', 'Losses'], height = bot_games['isWinner'].value_counts())
plt.title('Games against bots')
plt.xlabel('Result')
plt.savefig('plots/bot_analysis.png')
plt.show()


