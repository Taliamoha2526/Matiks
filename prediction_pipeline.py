import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
from model import Model   # your custom scratch logistic regression

class TrainingError(Exception):
    """Custom exception for training pipeline errors."""
    pass

def load_game_data(file_path: str) -> pd.DataFrame:
    """Load game history dataset safely with validation."""
    file = Path(file_path)
    if not file.exists():
        raise TrainingError(f"Dataset not found at {file}")

    df = pd.read_csv(file)
    if df.empty:
        raise TrainingError("Dataset is empty.")

    required_cols = {
        "gameId", "opponentId", "isWinner", "timestamp",
        "score", "opponentScore", "ratingChange", "duration",
        "gameType", "gameMode"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise TrainingError(f"Dataset missing required columns: {missing}")

    return df

def preprocess_data(df: pd.DataFrame):
    """Feature engineering + preprocessing pipeline with validation."""
    try:
        df['hour'] = pd.to_datetime(df['timestamp'], errors="coerce").dt.hour
        df['day'] = pd.to_datetime(df['timestamp'], errors="coerce").dt.dayofweek
    except Exception as e:
        raise TrainingError(f"Timestamp parsing failed: {e}")

    df['score_diff'] = df['score'] - df['opponentScore']

    # Labels must be array-like
    y = df['isWinner'].astype(int).values
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise TrainingError("Labels must be a 1D NumPy array.")

    # Features
    X = df.drop(['gameId', 'opponentId', 'isWinner', 'timestamp'], axis=1)

    categorical = ['gameType', 'gameMode']
    numerical = ['score', 'opponentScore', 'ratingChange', 'duration', 'hour', 'day']

    for col in categorical + numerical:
        if col not in X.columns:
            raise TrainingError(f"Missing feature column: {col}")

    preprocess = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical),
        ('num', StandardScaler(), numerical)
    ], remainder='passthrough')

    return X, y, preprocess

def train_model(X, y, preprocess, alpha=0.07, beta1=0.9, beta2=0.999, epochs=100):
    """Train logistic regression model with robust error handling."""
    if len(X) == 0 or len(y) == 0:
        raise TrainingError("Training data is empty.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    except Exception as e:
        raise TrainingError(f"Train/test split failed: {e}")

    try:
        X_train_scaled = preprocess.fit_transform(X_train)
        X_test_scaled = preprocess.transform(X_test)
    except Exception as e:
        raise TrainingError(f"Preprocessing failed: {e}")

    model = Model(alpha=alpha, epochs=epochs, beta1=beta1, beta2=beta2)
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        raise TrainingError(f"Model training failed: {e}")

    try:
        proba = np.array(model.predict_proba(X_test_scaled))
        if not isinstance(proba, (list, np.ndarray)):
            raise TrainingError("Model.predict_proba did not return array-like output.")
        pred = (np.array(proba) >= 0.5).astype(int)
    except Exception as e:
        raise TrainingError(f"Prediction failed: {e}")

    try:
        score = model.score(pred, y_test)
    except Exception as e:
        raise TrainingError(f"Scoring failed: {e}")

    return model, score
