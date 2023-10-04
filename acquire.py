import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

import config

api_key = config.ALPHA_VANTAGE_API_KEY

# ------------- Function to retrieve and process data for a stock symbol--------------

def get_stock_data(symbol, api_key):
    '''
    Acquisiiton function used to search for a csv if not present in os,
    then it will get the data for the previous 2 years and store it as a CSV.
    '''
    # Define the CSV file path for the stock symbol
    csv_file_path = f'{symbol}_data.csv'
    
    # Calculate the date 3.5 years ago from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Check if the CSV file exists and if it contains data within the desired time frame
    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
        
        # Check if the data covers the desired time frame
        if df.index[-1] >= start_date:
            # Filter the DataFrame to include only data within the desired time frame
            df = df[start_date:]
            return df
    
    # Fetch monthly data from the API for the desired time frame
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    
    monthly_data = data['Monthly Time Series']
    
    df = pd.DataFrame(monthly_data).T  # Transpose to have dates as index
    df = df.rename(columns={'1. open': f'{symbol}_open', '2. high': f'{symbol}_high', '3. low': f'{symbol}_low', '4. close': f'{symbol}_close', '5. volume': f'{symbol}_volume'})
    df.index = pd.to_datetime(df.index) # set the index for timestamps to datetime values; originally object.
    
    # Filter the DataFrame to include only data within the desired time frame
    df = df[start_date:]
    
    # Save the DataFrame as a CSV file for future use
    df.to_csv(csv_file_path)
    
    return df

# Get data by calling for NVDA, AAPL, and AMD
# nvda_data = get_stock_data('NVDA', api_key)
# aapl_data = get_stock_data('AAPL', api_key)
# amd_data = get_stock_data('AMD', api_key)

