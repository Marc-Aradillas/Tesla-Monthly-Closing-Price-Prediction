import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate import plot_residuals, regression_errors, baseline_mean_errors, better_than_baseline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from math import sqrt

#custom imports
import wrangle as w


# ------------------------ XY SPLIT FUNCTION ----------------------
# xy_split function to create usable subsets; reusable.
def xy_split(df, col):
    X = df.drop(columns=[col])
    y = df[col]
    return X, y


def eval_model(y_actual, y_hat):
    
    return sqrt(mean_squared_error(y_actual, y_hat))


def train_model(model, X_train, y_train, X_val, y_val):
    
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    
    train_rmse = eval_model(y_train, train_preds)
    
    val_preds = model.predict(X_val)
    
    val_rmse = eval_model(y_val, val_preds)
    
    print(f'The train RMSE is {train_rmse:.2f}.\n')
    print(f'The validate RMSE is {val_rmse:.2f}.\n\n')
    
    return model


# ------------------------------ Train and eval function -------------------------------------

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Train a machine learning model and evaluate its performance on training and validation data.
    
    Args:
        model (object): The machine learning model to be trained.
        X_train (array-like): Training feature data.
        y_train (array-like): Training target data.
        X_val (array-like): Validation feature data.
        y_val (array-like): Validation target data.
        
    Returns:
        object: The trained model.
        float: The training RMSE.
        float: The validation RMSE.
    """
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on training and validation data
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Evaluate the model's performance
    train_rmse = eval_model(y_train, train_preds)
    val_rmse = eval_model(y_val, val_preds)

    # Calculate R-squared (R2) for training and validation sets
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    
    # Print the results
    print(f"\n-------------------------------------")
    print(f'The train RMSE is {train_rmse:.2f}.\n')
    print(f"\n-------------------------------------")
    print(f'The validation RMSE is {val_rmse:.2f}.\n\n')
    print(f"\n-------------------------------------")
    print(f"\nTraining R-squared (R2): {train_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")
    
    return model, train_rmse, val_rmse

# Example usage:
# trained_model, train_rmse, val_rmse = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)


# --------------------------------- stock plot and model train -----------------------\

def lin_reg_baseline_model(df, target_column):

    # Filter out non-numeric columns (categorical features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    # Split data into X (features) and y (target)
    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]
    

    # Calculate baseline prediction (e.g., using the mean or median)
    # bl = y_train.mean()
    bl = y_train.median()

    # # Create a DataFrame to work with
    preds = pd.DataFrame({'y_actual': y_train, 'y_baseline': bl})

    # Calculate baseline residuals
    preds['y_baseline_residuals'] = bl - preds['y_actual']

    # Initialize and fit a linear regression model (you can use other models as needed)
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # Make predictions with the model
    preds['y_hat'] = lm.predict(X_train)

    # Calculate model residuals
    preds['y_hat_residuals'] = preds['y_hat'] - preds['y_actual']

    # Plot residuals (you may need to customize this function)
    plot_residuals(preds['y_actual'], preds['y_hat'])

    print("\n-------------------------------------")
    # Calculate regression errors
    mse = mean_squared_error(y_train, preds['y_hat'])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, preds['y_hat'])
    print(f"Model RMSE: {rmse:.2f}")
    print(f"Model R-squared: {r2:.2f}\n")
    print("-------------------------------------")

    # Calculate baseline errors
    mse_baseline = mean_squared_error(y_train, preds['y_baseline'])
    rmse_baseline = np.sqrt(mse_baseline)
    r2 = r2_score(y_train, preds['y_baseline'])
    print(f"Baseline RMSE: {rmse_baseline:.2f}")
    print(f"baseline R-squared: {r2:.2f}\n")
    print("-------------------------------------")

# ============================ model function =============================


def final_model_tsla(X_train, y_train, X_test, y_test):
    # Initialize the RandomForestRegressor
    rfr = RandomForestRegressor()
    
    # Train the model on the training data
    rfr.fit(X_train, y_train)
    
    # Make predictions on training and validation sets
    train_preds = rfr.predict(X_train)
    test_preds = rfr.predict(X_test)
    
    # Calculate RMSE for training and validation sets
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    # Calculate R-squared (R2) for training and validation sets
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    # Print the metrics
    print(f"\n-------------------------------------")
    print(f"\nTraining RMSE: {train_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTest RMSE: {test_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTraining R-squared (R2): {train_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTest R-squared (R2): {test_r2:.2f}")


def preprocess_data(df):
    """
    Preprocess the data by dropping unnecessary columns and splitting into features and target.
    """
    df = df.drop(columns=['month', 'day_of_week'])
    X, y = xy_split(df, 'next_month_close')
    return X, y

def train_random_forest_model(X_train, y_train, X_val, y_val):
    """
    Train a Random Forest Regressor model and evaluate it.
    """
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    train_preds = rfr.predict(X_train)
    val_preds = rfr.predict(X_val)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    print(f"\n-------------------------------------")
    print(f"\nTraining RMSE: {train_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation RMSE: {val_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTraining R-squared (R2): {train_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")

def train_linear_regression(train_df, val_df, target_column):
    """
    Train and evaluate a linear regression baseline model.
    """
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    train_df = train_df[numeric_cols]
    val_df = val_df[numeric_cols]
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]
    bl = y_train.median()
    train_preds = pd.DataFrame({'y_actual': y_train, 'y_baseline': bl})
    val_preds = pd.DataFrame({'y_actual': y_val, 'y_baseline': bl})
    train_preds['y_baseline_residuals'] = bl - train_preds['y_actual']
    val_preds['y_baseline_residuals'] = bl - val_preds['y_actual']
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    train_preds['y_hat'] = lm.predict(X_train)
    val_preds['y_hat'] = lm.predict(X_val)
    train_preds['y_hat_residuals'] = train_preds['y_hat'] - train_preds['y_actual']
    val_preds['y_hat_residuals'] = val_preds['y_hat'] - val_preds['y_actual']
    print("\nTraining Set Metrics:")
    evaluate_model(train_preds['y_actual'], train_preds['y_hat'])
    print("\nValidation Set Metrics:")
    evaluate_model(val_preds['y_actual'], val_preds['y_hat'])

def model_3(X_train, y_train, X_val, y_val, early_stopping_rounds=10, params=None):
    # Define the hyperparameters for your XGBoost model (or pass them as an argument)
    if params is None:
        params = {
            'learning_rate': 0.1,
            'n_estimators': 300,
            'max_depth': 4,
            'early_stopping_rounds': early_stopping_rounds,
            # Add other hyperparameters as needed
        }

    # Define weight data (you can replace this with your actual weights)
    sample_weights = np.ones(X_train.shape[0])  # Example: All weights are set to 1
    
    # Create the XGBoost regressor with your specified hyperparameters
    xgb = XGBRegressor(**params)
    
    # Fit the model to your training data with eval_set and verbose
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, sample_weight=sample_weights)

    # Access the best iteration and best score
    best_iteration = xgb.best_iteration
    best_score = xgb.best_score
    
    # Make predictions on validation set
    val_preds = xgb.predict(X_val)
    
    # Calculate RMSE and R2 for the validation set
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_r2 = r2_score(y_val, val_preds)

    # Make predictions on training set
    train_preds = xgb.predict(X_train)

    # Calculate RMSE and R2 for the training set
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_r2 = r2_score(y_train, train_preds)
    
    # Print the metrics within the function
    print(f"\n\n\n-------------------------------------")
    print(f"\nTraining RMSE: {train_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTraining R-squared (R2): {train_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation RMSE: {val_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nBest Score: {best_score}")


def evaluate_model(y_actual, y_pred):
    """
    Evaluate a regression model and print RMSE and R-squared.
    """
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")

# Your other module functions and imports here

# Define your model_3 function (as mentioned in the previous response)

# Your other module functions and code here

def main(tsla_train, tsla_val, tsla_test):
    # Load and preprocess your data (tsla_train, tsla_val, vvos_train, vvos_test)
    X_train, y_train = preprocess_data(tsla_train)
    X_val, y_val = preprocess_data(tsla_val)

    print('\nModel 1: Random Forest Regressor')
    train_random_forest_model(X_train, y_train, X_val, y_val)

    print('\n\n\n\nModel 2: Linear Regression Baseline')
    train_linear_regression(tsla_train, tsla_val, 'next_month_close')

    print('\n\n\n\nModel 3: XGBoost Regressor')
    # Assuming you have already loaded and preprocessed your data
    X_val, y_val = preprocess_data(tsla_val)  # You should have a function like this
    model_3_results = model_3(X_train, y_train, X_val, y_val)

    print('\n\n\n\nBest Model: Random Forest Regressor Model')
    X_train, y_train = preprocess_data(tsla_train)
    X_test, y_test = preprocess_data(tsla_test)
    final_model_tsla(X_train, y_train, X_test, y_test)

    # Repeat for vvos dataset if needed
    # X_train, y_train = preprocess_data(vvos_train)
    # X_val, y_val = preprocess_data(vvos_val)
    # train_random_forest_model(X_train, y_train, X_val, y_val)
    # train_linear_regression_baseline(vvos_train, vvos_val, 'next_month_close')
    # X_train, y_train = preprocess_data(vvos_train)
    # X_test, y_test = preprocess_data(vvos_test)
    # final_model_vvos(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main(tsla_train, tsla_val, tsla_test)

