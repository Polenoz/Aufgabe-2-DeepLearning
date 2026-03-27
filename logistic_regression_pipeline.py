from __future__ import annotations

import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Advertising.csv"
LOG_DIR = BASE_DIR / "logs"
TEST_DATA_DIR = BASE_DIR / "test_data"
FEATURE_COLUMNS = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
]
TARGET_COLUMN = "Clicked on Ad"


def _summarize_value(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return f"DataFrame(shape={value.shape})"
    if isinstance(value, pd.Series):
        return f"Series(shape={value.shape})"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape})"
    if isinstance(value, LogisticRegression):
        params = value.get_params()
        return (
            f"{value.__class__.__name__}(max_iter={params.get('max_iter')}, "
            f"solver={params.get('solver')})"
        )
    if isinstance(value, (list, tuple)) and len(value) > 8:
        return f"{type(value).__name__}(len={len(value)})"
    return repr(value)


def _get_logger(function_name: str) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(f"logistic_regression.{function_name}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / f"{function_name}.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def my_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = _get_logger(func.__name__)
        summarized_args = ", ".join(_summarize_value(arg) for arg in args)
        summarized_kwargs = ", ".join(
            f"{key}={_summarize_value(value)}" for key, value in kwargs.items()
        )
        logger.info("Called with args=[%s] kwargs=[%s]", summarized_args, summarized_kwargs)
        return func(*args, **kwargs)

    return wrapper


def my_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = _get_logger(func.__name__)
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start_time
        logger.info("Executed in %.6f seconds", runtime)
        return result

    return wrapper


class LoggedLogisticRegression(LogisticRegression):
    @my_logger
    @my_timer
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=sample_weight)

    @my_logger
    @my_timer
    def predict(self, X):
        return super().predict(X)


def load_advertising_data(data_path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def split_training_and_holdout_data(
    data: pd.DataFrame,
    test_size: float = 0.33,
    random_state: int = 42,
):
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_prediction_test_data(test_data_path: Path | str) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(test_data_path)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return X, y


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> tuple[float, np.ndarray]:
    accuracy = accuracy_score(y_true, predictions)
    matrix = confusion_matrix(y_true, predictions)
    logger = _get_logger("predict")
    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Confusion Matrix:\n%s", matrix)
    return accuracy, matrix


def build_default_model() -> LoggedLogisticRegression:
    return LoggedLogisticRegression(max_iter=1000)


def train_default_model(data_path: Path | str = DATA_PATH):
    data = load_advertising_data(data_path)
    X_train, X_test, y_train, y_test = split_training_and_holdout_data(data)
    model = build_default_model()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def measure_fit_runtime(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    repeats: int = 1,
) -> list[float]:
    durations = []
    for _ in range(repeats):
        model = build_default_model()
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        durations.append(time.perf_counter() - start_time)
    return durations


def benchmark_fit_runtime(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    repeats: int = 5,
) -> dict[str, float | list[float]]:
    durations = measure_fit_runtime(X_train, y_train, repeats=repeats)
    return {
        "durations": durations,
        "representative_runtime": max(durations),
    }


def save_expected_metrics(path: Path | str, accuracy: float, matrix: np.ndarray) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(exist_ok=True)
    payload = {
        "accuracy": accuracy,
        "confusion_matrix": matrix.tolist(),
    }
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_expected_metrics(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
