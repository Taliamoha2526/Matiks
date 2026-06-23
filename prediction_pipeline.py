import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
from model import Model


def load_game_data(file_path: str) -> pd.DataFrame:
    """Load game history dataset safely."""
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"Dataset not found at {file}")
    df = pd.read_csv(file)

    # Validate required columns
    required_cols = {"gameId", "opponentId", "isWinner", "timestamp",
                     "score", "opponentScore", "ratingChange", "duration",
                     "gameType", "gameMode"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns: {required_cols - set(df.columns)}")

    return df


def preprocess_data(df: pd.DataFrame):
    """Feature engineering + preprocessing pipeline."""
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['score_diff'] = df['score'] - df['opponentScore']

    y = df['isWinner']
    X = df.drop(['gameId', 'opponentId', 'isWinner', 'timestamp'], axis=1)

    categorical = ['gameType', 'gameMode']
    numerical = ['score', 'opponentScore', 'ratingChange', 'duration', 'hour', 'day']

    preprocess = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical),
        ('num', StandardScaler(), numerical)
    ], remainder='passthrough')

    return X, y, preprocess


def train_model(X, y, preprocess, alpha=0.07, beta1=0.9, beta2=0.999, epochs=100):
    """Train logistic regression model from scratch."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_scaled = preprocess.fit_transform(X_train)
    model = Model(alpha=alpha, epochs=epochs, beta1=beta1, beta2=beta2)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = preprocess.transform(X_test)
    pred = [True if x >= 0.5 else False for x in model.predict_proba(X_test_scaled)]
    score = model.score(pred, y_test)

    return model, score




