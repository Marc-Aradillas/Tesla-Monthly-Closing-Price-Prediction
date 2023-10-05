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
def xy_splt():
# Wrangle the data
    nvda_df, aapl_df, amd_df, tsla_df, vvos_df = w.wrangle_stock_data()
    
    # Drop categorical features
    vvos_df = vvos_df.drop(columns=['day_of_week', 'month'])

    # Train-test split
    train, val, test = train_val_test(vvos_df)

    # Split data into X and y for train and val
    X_train, y_train = xy_split(train, 'home_value')
    X_val, y_val = xy_split(val, 'home_value')
    X_test, y_test = xy_split(test, 'home_value')

    return X_train, y_train, X_val, y_val, X_test, y_test


def model_1(X_train, y_train, X_val, y_val):
    # Initialize the RandomForestRegressor
    rfr = RandomForestRegressor()
    
    # Train the model on the training data
    rfr.fit(X_train, y_train)
    
    # Make predictions on training and validation sets
    train_preds = rfr.predict(X_train)
    val_preds = rfr.predict(X_val)
    
    # Calculate RMSE for training and validation sets
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    # Calculate R-squared (R2) for training and validation sets
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    
    # Print the metrics
    print(f"\n-------------------------------------")
    print(f"\nTraining RMSE: {train_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation RMSE: {val_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTraining R-squared (R2): {train_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")
    
    return rfr

# Example usage:
# rfr = RandomForestRegressor()
# trained_model = train_and_evaluate_model(rfr, X_train, y_train, X_val, y_val)

# ======================================= model 2 ============================================\


def model_2(df, target_column, X_val, y_val, early_stopping_rounds=10, params=None):

    # acquire data
    df = wrangle_zillow()

    # Drop categorical features
    df = df.drop(columns=['property_county_landuse_code', 'property_zoning_desc', 'n-prop_type', 'n-av_room_size', 'state'])
    
    # Train-test split
    train, val, test = train_val_test(df)

    # Split data into X and y
    X_train, y_train = xy_split(df, target_column)
    
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
    
    # Create a dictionary to store the results
    results = {
        'model': xgb,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'best_iteration': best_iteration,
        'best_score': best_score
    }
    
    # Print the metrics within the function
    print(f"\n\n\n-------------------------------------")
    print(f"\nValidation RMSE: {val_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nBest Iteration: {best_iteration}")
    print(f"\n-------------------------------------")
    print(f"\nBest Score: {best_score}")
    
    return results


# ========================================= model 3 ================================================

def model_3(X_train, y_train, X_val, y_val):

        # Calculate mean and median of y_train
        y_train_mean = y_train.mean()
        y_train_median = y_train.median()
        
        # Create a DataFrame with y_train statistics
        bl = pd.DataFrame({"y_actual" : y_train,
                           "y_mean" : y_train_mean,
                           "y_median" : y_train_median})
        
        # Apply polynomial feature transformation
        poly = PolynomialFeatures()
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val)
    
        # Train a Linear Regression model and evaluate it
        lm = LinearRegression()
        trained_model, train_rmse, val_rmse = train_and_evaluate_model(lm, X_train, y_train, X_val, y_val)
    
        # Print the metrics
        print(f"\n-------------------------------------")
        print(f"\nTraining RMSE: {train_rmse:.2f}")
        print(f"\n-------------------------------------")
        print(f"\nValidation RMSE: {val_rmse:.2f}")
        print(f"\n-------------------------------------")
        print(f"\nTraining R-squared (R2): {train_r2:.2f}")
        print(f"\n-------------------------------------")
        print(f"\nValidation R-squared (R2): {val_r2:.2f}")

# ======================== final model and visualization ===============================
def final_model(df, target_column, X_test, y_test, early_stopping_rounds=10, params=None):
    
    # acquire data
    df = wrangle_zillow()

    # Drop categorical features
    df = df.drop(columns=['property_county_landuse_code', 'property_zoning_desc', 'n-prop_type', 'n-av_room_size', 'state'])
    
    # Train-test split
    train, val, test = train_val_test(df)

    # Split data into X and y
    X_train, y_train = xy_split(df, target_column)
    X_test, y_test = xy_split(df, target_column)
    
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
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False, sample_weight=sample_weights)

    # Access the best iteration and best score
    best_iteration = xgb.best_iteration
    best_score = xgb.best_score
    
    # Make predictions on validation set
    test_preds = xgb.predict(X_test)

    # Create a scatter plot of actual vs. predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=test_preds, alpha=0.5, color='orange')
    plt.title("Actual vs. Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()
    
    # Calculate RMSE and R2 for the validation set
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)
    
    # Create a dictionary to store the results
    results = {
        'model': xgb,
        'val_rmse': test_rmse,
        'val_r2': test_r2,
        'best_iteration': best_iteration,
        'best_score': best_score
    }
    
    # Print the metrics within the function
    print(f"\n\n\n-------------------------------------")
    print(f"\nTest RMSE: {test_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTest R-squared (R2): {test_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nBest Iteration: {best_iteration}")
    print(f"\n-------------------------------------")
    print(f"\nBest Score: {best_score}")
    
    return results
