# Logistic Regression Aufgabe 2

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Polenoz/Polenoz-logistic-regression-aufgabe-2/main?labpath=run_tests.ipynb)

This repository is prepared as a separate `Prüfungsaufgabe 2` submission for the logistic regression exercise from the course `Python für Data Science, Maschinelles Lernen & Visualization`. It extends the original example with automated testing, logging, and runtime checks.

## Project Contents

- `Advertising.csv`: dataset used by the project
- `logistic_regression_pipeline.py`: reusable Python pipeline with `my_logger` and `my_timer`
- `tests/test_logistic_regression.py`: Python `unittest` test cases for `predict()` and `fit()`
- `test_data/predict_test_data.csv`: holdout test data file used by the prediction test
- `test_data/expected_metrics.json`: expected accuracy and confusion matrix
- `run_tests.ipynb`: Binder notebook for running the two tests
- `runtime.txt`: Python runtime definition for Binder
- `requirements.txt`: required Python packages

## What the Project Does

The project trains a logistic regression model on the Advertising dataset and checks two things with automated tests:

1. `predict()` produces the expected classification quality on the saved test dataset.
2. `fit()` stays within 120% of a representative training runtime.

The logging decorators write function calls and runtimes to the `logs/` directory:

- `logs/fit.log`
- `logs/predict.log`

## How to Run the Tests With Binder

1. Click the Binder badge above.
2. Wait until the Binder environment is built and opened.
3. Open `run_tests.ipynb` if it is not already open.
4. Run all cells from top to bottom.
5. The first code cell runs both required `unittest` test cases.
6. The second code cell prints the generated log files from `logs/fit.log` and `logs/predict.log`.

## How to Run the Automated Tests

### Local

Run the two required `unittest` test cases step by step:

1. Create and activate the project environment.
2. Install the dependencies from `requirements.txt`.
3. Run the test suite from the repository root.

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

The executed test cases are:

- `test_predict_outputs_expected_accuracy_and_confusion_matrix`
- `test_fit_runtime_stays_within_120_percent_of_representative_runtime`

### Google Colab

If you still want to execute the tests in Colab, run the following commands in a Colab cell:

```python
!git clone https://github.com/Polenoz/Polenoz-logistic-regression-aufgabe-2.git
%cd Polenoz-logistic-regression-aufgabe-2
!pip install -r requirements.txt
!python -m unittest discover -s tests -v
```

## Test Data File

The prediction test explicitly loads the separate test data file `test_data/predict_test_data.csv`.

Location:

```text
test_data/predict_test_data.csv
```

This file contains the holdout dataset with the following columns:

- `Daily Time Spent on Site`
- `Age`
- `Area Income`
- `Daily Internet Usage`
- `Male`
- `Clicked on Ad`

## Expected Results

When `test_predict_outputs_expected_accuracy_and_confusion_matrix` reads `test_data/predict_test_data.csv`, it should reproduce these metrics:

```text
Accuracy: 0.9667
Confusion Matrix:
[[158   4]
 [  7 161]]
```

When `test_fit_runtime_stays_within_120_percent_of_representative_runtime` runs, the project measures a representative `fit()` runtime from repeated runs and verifies that a new training run stays below `120%` of that value.

## Documented Screen Output For Both Tests

The following example output was produced by running the two required tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

```text
test_fit_runtime_stays_within_120_percent_of_representative_runtime ... ok
test_predict_outputs_expected_accuracy_and_confusion_matrix ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.463s

OK
Representative fit runtime: 0.070844 seconds
Measured fit runtime: 0.052630 seconds
Runtime limit (120%): 0.085013 seconds
Accuracy: 0.9667
Confusion Matrix:
[[158   4]
 [  7 161]]
```

This output documents both required test cases separately. The accuracy and confusion matrix should match exactly; the runtime values can vary slightly between environments.

## Notes on the Implementation

`logistic_regression_pipeline.py` includes two decorators based on the approach described in the article `Unit Testing and Logging for Data Science`:

- `my_logger`: logs the function name and summarized input arguments
- `my_timer`: logs the runtime of the wrapped function

These decorators are applied to the model methods `fit()` and `predict()`.
