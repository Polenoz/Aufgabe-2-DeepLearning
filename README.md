# Logistic Regression Testing Project

In this project, a logistic regression model is trained on the Advertising dataset.  
The goal is not only to build the model, but also to test its behavior and document the training process with logging.

## Dataset

The project uses the file:

- Advertising.csv

Make sure the file is in the same directory as the notebook.

## Required libraries

This project uses the following libraries:

- pandas
- numpy
- scikit-learn
- logging
- time

## How to run

You can run the notebook in Google Colab:

- Open the notebook
- Upload the dataset file
- Run all cells

You can also run it locally:

- Install the required libraries
- Open the notebook with Jupyter
- Run all cells

Example installation:

pip install pandas numpy scikit-learn jupyter

## What is done in this notebook

- The dataset is loaded and prepared
- A logistic regression model is created
- The model is trained
- Predictions are made
- Accuracy, confusion matrix and classification report are printed
- Logging is added with a custom logger
- Runtime measurement is added with a custom timer
- Two tests are performed:
  - predict() test
  - fit() runtime test

## Tests

### predict() test
This test checks whether the prediction accuracy is above a defined threshold.

### fit() runtime test
This test checks whether model training finishes within an acceptable time limit.
