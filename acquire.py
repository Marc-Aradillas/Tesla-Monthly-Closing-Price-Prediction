import pandas as pd
import numpy as np
import requests
import os

import config

api_key = config.ALPHA_VANTAGE_API_KEY

# ------------- Function to retrieve and process data for a stock symbol--------------
def get_stock_data(symbol, api_key):
    '''
    Acquisiiton function used ot search for a csv if not present in os, then it will get the data and store as csv.
    '''
    # Define the CSV file path for the stock symbol
    csv_file_path = f'{symbol}_data.csv'
    
    # Check if the CSV file exists
    if os.path.isfile(csv_file_path):
        # If the CSV file exists, read data from the CSV
        df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
    else:
        # If the CSV file doesn't exist, fetch data from the API
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}'
        r = requests.get(url)
        data = r.json()
        
        daily_data = data['Time Series (Daily)']
        
        df = pd.DataFrame(daily_data).T  # Transpose to have dates as index
        df = df.rename(columns={'1. open': f'{symbol}_open', '2. high': f'{symbol}_high', '3. low': f'{symbol}_low', '4. close': f'{symbol}_close', '5. volume': f'{symbol}_volume'})
        df.index = pd.to_datetime(df.index) # set the index for timestamps to datetime values; originally object.
        
        # Save the DataFrame as a CSV file for future use
        df.to_csv(csv_file_path)
    
    return df


# Get data by calling for NVDA, AAPL, and AMD
# nvda_data = get_stock_data('NVDA', api_key)
# aapl_data = get_stock_data('AAPL', api_key)
# amd_data = get_stock_data('AMD', api_key)

