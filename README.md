
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


## Output

```
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/dense.py:106: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Calling fit_model...
fit_model took 27.0125 seconds.
fit_model finished.
Calling predict_model...
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
predict_model finished.

Accuracy: 0.97
Confusion Matrix:
[[ 970    0    0    1    0    3    3    1    2    0]
 [   0 1118    3    2    0    2    3    1    6    0]
 [   9    3  994    3    1    1    6    8    6    1]
 [   1    0    5  968    0   15    1    5    6    9]
 [   1    0    4    1  956    0    6    3    3    8]
 [   2    0    0    8    2  865    6    1    5    3]
 [   4    2    4    1    5    6  934    0    2    0]
 [   1    7    7    4    3    0    0  989    3   14]
 [   3    0    3    4    2    3    4    6  945    4]
 [   2    2    0    3   13    4    0    8    3  974]]
Calling fit_model...
fit_model took 12.2019 seconds.
fit_model finished.

Reference fit time: 27.0125s, Current fit time: 12.2019s
Calling fit_model...
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/dense.py:106: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
test_fit_time (__main__.TestDeepLearningModel.test_fit_time) ... 
fit_model took 15.2363 seconds.
fit_model finished.
Calling fit_model...
ok
test_predict_accuracy (__main__.TestDeepLearningModel.test_predict_accuracy) ... 
fit_model took 12.0092 seconds.
fit_model finished.

Reference fit time: 15.2363s, Current fit time: 12.0092s
Calling predict_model...
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
predict_model finished.
ok

----------------------------------------------------------------------
Ran 2 tests in 28.049s

OK

Accuracy: 0.97
Confusion Matrix:
[[ 966    0    3    2    0    2    2    2    3    0]
 [   0 1124    5    0    0    0    1    1    4    0]
 [   1    1 1000    7    2    1    5    6    7    2]
 [   0    0    5  989    0    4    0    5    4    3]
 [   0    0    3    2  954    0    9    3    1   10]
 [   2    0    0   11    1  865    2    2    5    4]
 [   4    2    3    1    1    5  938    0    4    0]
 [   2    1   11    3    1    1    0  999    2    8]
 [   1    1    2    6    5    9    1    3  943    3]
 [   2    1    0    6    7    4    0   10    8  971]]
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
