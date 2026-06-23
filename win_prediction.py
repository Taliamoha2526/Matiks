import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from model import Model

game_data = pd.read_csv('data/gameHistory_games.csv')
game_data['hour'] = pd.to_datetime(game_data['timestamp']).dt.hour
game_data['day'] = pd.to_datetime(game_data['timestamp']).dt.dayofweek
game_data['score_diff'] = game_data['score'] - game_data['opponentScore']
Y = game_data['isWinner']
X = game_data = game_data.drop(['gameId', 'opponentId', 'isWinner', 'timestamp'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)


categorical = ['gameType', 'gameMode']
numerical = ['score', 'opponentScore', 'ratingChange', 'duration', 'hour', 'day']

preprocess = ColumnTransformer([('cat', OneHotEncoder(), categorical),
                                ('num', StandardScaler(), numerical)],
                               remainder = 'passthrough')
X_scaled = preprocess.fit_transform(X_train)
log = Model(alpha = 0.07, tries= 100)
log.fit(X_scaled, y_train)


X_test_scaled = preprocess.transform(X_test)
pred = [True if x >=0.5 else False for x in log.predict_log(X_test_scaled)]
print(log.score(pred, y_test))



