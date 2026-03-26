from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from logistic_regression_pipeline import (
    TEST_DATA_DIR,
    benchmark_fit_runtime,
    evaluate_predictions,
    load_advertising_data,
    load_expected_metrics,
    load_prediction_test_data,
    split_training_and_holdout_data,
    train_default_model,
)


EXPECTED_METRICS_PATH = TEST_DATA_DIR / "expected_metrics.json"
PREDICTION_TEST_DATA_PATH = TEST_DATA_DIR / "predict_test_data.csv"


class LogisticRegressionSystemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.data = load_advertising_data(cls.project_root / "Advertising.csv")
        cls.X_train, _, cls.y_train, _ = split_training_and_holdout_data(cls.data)
        cls.model, _, _, _, _ = train_default_model(cls.project_root / "Advertising.csv")
        cls.expected_metrics = load_expected_metrics(EXPECTED_METRICS_PATH)

    def test_predict_outputs_expected_accuracy_and_confusion_matrix(self):
        X_test, y_test = load_prediction_test_data(PREDICTION_TEST_DATA_PATH)
        predictions = self.model.predict(X_test)
        accuracy, matrix = evaluate_predictions(y_test, predictions)

        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(matrix)

        self.assertAlmostEqual(
            accuracy,
            self.expected_metrics["accuracy"],
            places=4,
        )
        np.testing.assert_array_equal(
            matrix,
            np.array(self.expected_metrics["confusion_matrix"]),
        )

    def test_fit_runtime_stays_within_120_percent_of_representative_runtime(self):
        benchmark = benchmark_fit_runtime(self.X_train, self.y_train, repeats=5)
        representative_runtime = benchmark["representative_runtime"]
        runtime_limit = representative_runtime * 1.2
        actual_runtime = benchmark_fit_runtime(self.X_train, self.y_train, repeats=1)[
            "representative_runtime"
        ]

        print(f"Representative fit runtime: {representative_runtime:.6f} seconds")
        print(f"Measured fit runtime: {actual_runtime:.6f} seconds")
        print(f"Runtime limit (120%): {runtime_limit:.6f} seconds")

        self.assertLessEqual(actual_runtime, runtime_limit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
