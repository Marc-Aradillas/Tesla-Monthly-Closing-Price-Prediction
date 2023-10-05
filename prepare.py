# imported libs for scaling
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split

#custom import
import acquire as a

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


# clean and save call
# cleaned_nvda_df = prep_and_save(nvda_data, 'NVDA')


# -----------------Train-Validate-Test-------------------------------

def train_val_test(df, seed=42):
    """
    Split the data into training, validation, and test sets.

    Parameters:
    - df (DataFrame): The input DataFrame to be split.
    - seed (int): Random seed for reproducibility.

    Returns:
    - train, val, test (DataFrames): Split datasets for training, validation, and testing.
    """
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)

    return train, val, test
