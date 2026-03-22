
# Deep Learning Unit Test Automation (Aufgabe 2) — MNIST Version

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Polenoz/Aufgabe-2-DeepLearning/blob/main/Aufgabe2_DeepLearning_Colab.ipynb)

This project demonstrates automated testing and logging for a deep learning model using the MNIST dataset. It implements logging and timing decorators, and provides unittests for the model's fit() and predict() functions. No data upload is required; MNIST is loaded directly in the notebook.


## Contents
- `Aufgabe2_DeepLearning_Colab.ipynb`: Colab notebook with all code, logging, and tests
- `requirements.txt`: Required Python packages (for local use)


## How to Run

### Google Colab (Recommended)
1. Click the Colab badge above or open the notebook in Colab.
2. Run all cells from top to bottom. No data upload is needed.
3. The notebook will train a model, run both required unittests, and show all outputs (accuracy, confusion matrix, timing, and test results).

### Local (Jupyter, also possible)
1. Clone or download this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook and run all cells.


## Example Output

```
test_fit_time (__main__.TestDeepLearningModel) ... 
Reference fit time: 8.1234s, Current fit time: 7.9123s
ok
test_predict_accuracy (__main__.TestDeepLearningModel) ... 
Accuracy: 0.97
Confusion Matrix:
[[ 972    1    0 ...]
 ...]
ok

----------------------------------------------------------------------
Ran 2 tests in 16.04s

OK
```

### How to Interpret the Results

- If both tests show 'ok' and accuracy is above 0.9, your model and tests are working as expected.
- If you see ImportError, run the first cell to install missing packages in Colab.
- If accuracy is below 0.9 or the fit time test fails, try rerunning the notebook or check your Colab runtime.

### Troubleshooting

- Make sure you are running in Google Colab for best compatibility.
- If you encounter errors, restart the runtime and run all cells again.


## License
This project is for educational purposes.
