import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures

# ----------------------- WRANGLE STOCKS-----------------------
# wrapping Acquire and Prep functions into one for wrangle.

import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

import config

api_key = config.ALPHA_VANTAGE_API_KEY


def get_stock_data(symbol, api_key):
    '''
    Acquisiiton function used to search for a csv if not present in os,
    then it will get the data for the previous 3 years and store it as a CSV.
    '''
    # Define the CSV file path for the stock symbol
    csv_file_path = f'{symbol}_data.csv'
    
    # Calculate the date 3 years ago from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    # Check if the CSV file exists
    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
        
        # Check if the data contains any records within the desired time frame
        if not df.empty and df.index[-1] >= start_date:
            # Filter the DataFrame to include only data within the desired time frame
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            return df
    
    # Fetch monthly data from the API for the desired time frame
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    
    # Check if the API response contains an error message
    if 'Error Message' in data:
        raise ValueError(f"API Error: {data['Error Message']}")
    
    monthly_data = data['Monthly Time Series']
    
    df = pd.DataFrame(monthly_data).T  # Transpose to have dates as index
    df = df.rename(columns={'1. open': f'{symbol}_open', '2. high': f'{symbol}_high', '3. low': f'{symbol}_low', '4. close': f'{symbol}_close', '5. volume': f'{symbol}_volume'})
    df.index = pd.to_datetime(df.index) # set the index for timestamps to datetime values; originally object.
    
    # Filter the DataFrame to include only data within the desired time frame
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # Save the DataFrame as a CSV file for future use
    df.to_csv(csv_file_path)
    
    return df



def prep(df, symbol, csv_file_name=None):
    '''
    Prepare function used to take a specific stock company dataframe, clean the information, 
    create a couple of features, and optionally save it as a CSV file in the current working directory.
    
    Args:
        df (pd.DataFrame): The dataframe to be prepared.
        symbol (str): The symbol or identifier for the stock.
        csv_file_name (str, optional): The name to save the clean dataframe as a CSV file. 
                                      If None, the dataframe is not saved.
    
    Returns:
        pd.DataFrame: The cleaned dataframe.
    '''
    
    # Clean and preprocess data

    df = df.apply(pd.to_numeric)
    df.columns = [col.lower() for col in df.columns]
    df['month'] = df.index.month_name()
    df['day_of_week'] = df.index.day_name()
    df['year'] = df.index.year

    
    # Save the cleaned dataframe as a CSV file in the current working directory with the symbol in the file name if a file name is provided
    if csv_file_name is not None:
        csv_file_name = f'{symbol}_{csv_file_name}.csv'
        df.to_csv(csv_file_name)
    
    return df

def wrangle_stock_data():
    # Get data for NVDA, AAPL, and AMD
    nvda_data = get_stock_data('NVDA', api_key)
    aapl_data = get_stock_data('AAPL', api_key)
    amd_data = get_stock_data('AMD', api_key)
    tsla_data = get_stock_data('TSLA', api_key)
    vvos_data = get_stock_data('VVOS', api_key)
    
    # Clean and prepare the data
    nvda_df = prep(nvda_data, 'NVDA', 'cleaned_data')
    aapl_df = prep(aapl_data, 'AAPL', 'cleaned_data')
    amd_df = prep(amd_data, 'AMD', 'cleaned_data')
    tsla_df = prep(tsla_data, 'TSLA', 'cleaned_data')
    vvos_df = prep(vvos_data, 'VVOS' 'cleaned_data')

    
    # Add 'next_day_close' column to each dataframe #
    # target_column
    nvda_df['next_month_close'] = nvda_df['nvda_close'].shift(-1)
    aapl_df['next_month_close'] = aapl_df['aapl_close'].shift(-1)
    amd_df['next_month_close'] = amd_df['amd_close'].shift(-1)
    tsla_df['next_month_close'] = tsla_df['tsla_close'].shift(-1)
    vvos_df['next_month_close'] = vvos_df['vvos_close'].shift(-1)

    
    
    # Drop the last row to remove NaN values
    nvda_df = nvda_df[:-1]
    aapl_df = aapl_df[:-1]
    amd_df = amd_df[:-1]
    tsla_df = tsla_df[:-1]
    vvos_df = vvos_df[:-1]
    
    # Return the cleaned dataframes (and optionally train, validate, and test sets)
    return nvda_df, aapl_df, amd_df, tsla_df, vvos_df

# Example usage:
# nvda_df, aapl_df, amd_df = wrangle_stock_data()



# ------------------------ SPLIT FUNCTION -------------------------
# train val test split function
def train_val_test(df, seed = 42):
    """
    splits cleaned df into train, validate, and split
    
    Returns:
    - train, validate, split subset of df (dataframe): Splitted Wrangled Zillow Data
    """
    # data is split into 70% train and from the 30%, 50% goes to each test and validate subsets.
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed)

    
    return train, val, test


# ------------------------ SPLIT/SCALE FUNCTION -------------------------
# train val test split/scale function
def split_and_scale_data(df, seed=42):
    """
    Splits the data into train, validate, and test sets, and scales all numerical features.
    
    Parameters:
    - df (dataframe): The input dataframe.
    - seed (int): Random seed for reproducibility.

    Returns:
    - train_scaled, validate_scaled, test_scaled (dataframes): Scaled train, validate, and test sets.
    """

    # Split the data into train, validate, and test sets
    train, val, test = train_val_test(df, seed)

    # Extract the numerical features to scale (assuming all non-categorical columns are numerical)
    numerical_features = [col for col in df.select_dtypes(include=['number']).columns if col != 'year']

    # Scale all numerical features using standardscale scaling
    scaler = MinMaxScaler()
    scaler.fit(train[numerical_features])
    train[numerical_features] = scaler.transform(train[numerical_features])
    val[numerical_features] = scaler.transform(val[numerical_features])
    test[numerical_features] = scaler.transform(test[numerical_features])

    return train, val, test


# ------------------------ XY SPLIT FUNCTION ----------------------
# xy_split function to create usable subsets; reusable.
def xy_split(df, col):
    X = df.drop(columns=[col])
    y = df[col]
    return X, y


# ------------------------ XY SPLIT TVT FUNCTION ----------------------
def scale_data(train, val, test, to_scale):
    # make copies for scaling
    train_scaled = train.copy()
    validate_scaled = val.copy()
    test_scaled = test.copy()

    # scaling tool
    scaler = StandardScaler()

    #fit train set
    scaler.fit(train[to_scale])

    # transform the set
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(val[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled
