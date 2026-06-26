import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
    X = df.drop(['gameId', 'opponentId', 'isWinner', 'timestamp'], axis=1, errors='ignore')
    categorical = ['gameType', 'gameMode']
    numerical = ['score', 'opponentScore', 'ratingChange', 'duration', 'hour', 'day']

    for col in categorical + numerical:
        if col not in X.columns:
            raise TrainingError(f"Missing feature column: {col}")

    preprocess = ColumnTransformer([
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) ]), categorical),
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())]), numerical)], remainder='drop')

    return X, y, preprocess

def train_model(X, y, preprocess, alpha=0.001, beta1=0.9, beta2=0.999, epochs=500):
    """Train logistic regression model with robust error handling."""
    if len(X) == 0 or len(y) == 0:
        raise TrainingError("Training data is empty.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)
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
        score = model.score(X_test_scaled, y_test)
    except Exception as e:
        raise TrainingError(f"Scoring failed: {e}")

    return model, score, model.history, model.losses

def save_pipeline_artifacts(model, accuracy, history_acc, history_loss, base_dir="artifacts"):
    """
    Saves all model training artifacts (weights, histories, plots) into organized directories.

    Parameters:
        model: Trained Model instance containing .weights and .bias
        accuracy (float): Final test accuracy score
        history_acc (list): List of accuracies per epoch
        history_loss (list): List of losses per epoch
        base_dir (str/Path): The root folder where folders will be organized
    """
    root_path = Path(base_dir)
    models_dir = root_path / "models"
    metrics_dir = root_path / "metrics"
    plots_dir = root_path / "plots"

    # 2. Safely create directories if they do not exist
    for directory in [models_dir, metrics_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    try:
        # 3. Save Model Weights & Biases as Binary Numpy Files
        if model.weights is not None:
            np.save(models_dir / "model_weights.npy", model.weights)
            np.save(models_dir / "model_bias.npy", np.array([model.bias]))

        # 4. Save Final Performance and History Metrics
        # Saving as text/csv makes it easy to read later without loading Python
        np.savetxt(metrics_dir / "history_accuracies.csv", history_acc, delimiter=",", header="accuracy")
        np.savetxt(metrics_dir / "history_losses.csv", history_loss, delimiter=",", header="loss")

        # Save summary metadata summary
        summary = {"final_test_accuracy": float(accuracy), "epochs_trained": len(history_loss)}
        with open(metrics_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        # 5. Generate and Save Dual Convergence Plot
        epochs = range(1, len(history_loss) + 1)
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Left axis: Loss
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color="crimson")
        ax1.plot(epochs, history_loss, color="crimson", linewidth=2, label="Train Loss")
        ax1.tick_params(axis="y", labelcolor="crimson")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Right axis: Accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color="teal")
        ax2.plot(epochs, history_acc, color="teal", linewidth=2, linestyle="-.", label="Train Accuracy")
        ax2.tick_params(axis="y", labelcolor="teal")

        plt.title(f"Training Metrics Convergence (Final Test Acc: {accuracy:.2%})")
        fig.tight_layout()

        # Save figure to file
        plt.savefig(plots_dir / "training_convergence.png", dpi=300)
        plt.close()  # Close to free memory cache

        print("All artifacts saved successfully.")

    except Exception as e:
        print(f" Failed to save artifacts: {e}")
