import pandas as pd
import os

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
    
    # Save the cleaned dataframe as a CSV file in the current working directory with the symbol in the file name if a file name is provided
    if csv_file_name is not None:
        csv_file_name = f'{symbol}_{csv_file_name}.csv'
        df.to_csv(csv_file_name)
    
    return df


# clean and save call
# cleaned_nvda_df = prep_and_save(nvda_data, 'NVDA')
