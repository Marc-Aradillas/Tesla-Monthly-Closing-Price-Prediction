# imported libs
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from math import sqrt

#----------------------------- PLOT RESDIDS FUNCTION ----------------------------

# created residual plot
def plot_residuals(y, yhat):
    """
    a residual plot.

    Parameters:
    - y: Actual values
    - yhat: Predicted values
    """
    residuals = yhat - y
    plt.figure(figsize=(10, 6))
    plt.scatter(y, residuals, alpha=0.75, color='orange') # should be y because i want to plot actuals
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Model Residual Plot")
    plt.axhline(0, color='darkred', linestyle='dashdot')
    plt.tight_layout()
    plt.show()

#----------------------------- REG ERR FUNCTION ----------------------------

# returns values
def regression_errors(y, yhat):
    """
    Returns various regression error metrics.

    Parameters:
    - y: Actual values
    - yhat: Predicted values

    Returns:
    - SSE: Sum of Squared Errors
    - ESS: Explained Sum of Squares
    - TSS: Total Sum of Squares
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    """
    SSE = np.sum((y - yhat) ** 2)
    ESS = np.sum((yhat - np.mean(y)) ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)
    MSE = SSE / len(y)
    RMSE = np.sqrt(MSE)
    
    return SSE, ESS, TSS, MSE, RMSE


#---------------------- BASELINE MEAN ERR FUNCTION -------------------

# computes the SSE, MSE, and RMSE for the baseline model
def baseline_mean_errors(y):
    """
    Computes the SSE, MSE, and RMSE for the baseline model.

    Parameters:
    - y: Actual values

    Returns:
    - SSE: Sum of Squared Errors for the baseline
    - MSE: Mean Squared Error for the baseline
    - RMSE: Root Mean Squared Error for the baseline
    """
    y_baseline = np.mean(y)
    SSE_baseline = np.sum((y - y_baseline) ** 2)
    MSE_baseline = SSE_baseline / len(y)
    RMSE_baseline = np.sqrt(MSE_baseline)
    
    return SSE_baseline, MSE_baseline, RMSE_baseline


#---------------------- BASELINE MEAN ERR FUNCTION -------------------

# returns true if your model performs better than the baseline, otherwise false
def better_than_baseline(y, yhat):
    """
    Returns True if the model performs better than the baseline, otherwise False.

    Parameters:
    - y: Actual values
    - yhat: Predicted values

    Returns:
    - True if model's RMSE < baseline's RMSE, else False
    """
    _, _, _, _, RMSE = regression_errors(y, yhat)
    _, _, RMSE_baseline = baseline_mean_errors(y)
    
    return RMSE < RMSE_baseline




# _________________________________________________________________________

# evaluate.plot_residuals(preds.y_actual, preds.y_hat)

# print(f"\n-------------------------------------")


# SSE, ESS, TSS, MSE, RMSE = evaluate.regression_errors(preds.y_actual, preds.y_hat)

# print(f"\nSSE: {SSE}\n")

# print(f"ESS: {ESS}\n")

# print(f"TSS: {TSS}\n")

# print(f"MSE: {MSE}\n")

# print(f"RMSE: {RMSE}\n")

# print(f"\n-------------------------------------")

# SSE_baseline, MSE_baseline, RMSE_baseline = evaluate.baseline_mean_errors(preds.y_actual)

# print(f"Baseline SSE: {SSE_baseline}\n")

# print(f"Baseline MSE: {MSE_baseline}\n")

# print(f"Baseline RMSE: {RMSE_baseline}\n")

# print(f"\n-------------------------------------")

# print(f"\nIs the model better than the baseline? {evaluate.better_than_baseline(preds.y_actual, preds.y_hat)}")
