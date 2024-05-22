import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Replace 'your_api_key_here' with your actual CryptoCompare API key
API_KEY = ""

# Function to get Ethereum data from CryptoCompare API in batches
def get_eth_data(api_key, limit=2000):
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    end_time = int(time.time())
    all_data = []
    
    while len(all_data) < 6 * 30 * 24:  # Approximate hours in 6 months
        params = {
            'fsym': 'ETH',
            'tsym': 'USD',
            'limit': limit,
            'toTs': end_time,
            'api_key': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['Response'] == 'Success':
            data = data['Data']['Data']
            all_data.extend(data)
            end_time = data[0]['time'] - 1  # Update the end_time to the timestamp of the earliest data point
        else:
            print("Error fetching data:", data['Message'])
            break
    
    df = pd.DataFrame(all_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volumeto': 'Volume'}, inplace=True)
    df['Unix Timestamp'] = df['Date'].astype(int) // 10**9 * 1000  # Convert to Unix timestamp in milliseconds
    df['Symbol'] = 'ETHUSD'
    df = df[['Unix Timestamp', 'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Exclude the first 10 and the last row
    df = df.iloc[10:-1]
    
    return df

# Function to get the latest Ethereum price from CryptoCompare API
def get_latest_eth_price(api_key):
    url = "https://min-api.cryptocompare.com/data/price"
    params = {
        'fsym': 'ETH',
        'tsyms': 'USD',
        'api_key': api_key
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'USD' in data:
        return data['USD']
    else:
        print("Error fetching latest price:", data)
        return None

# Fetch the historical data
eth_data = get_eth_data(API_KEY)
if eth_data is not None:
    print("Last 6 months of hourly Ethereum data fetched.")
    print(eth_data.tail())  # Display the last few rows of the data

    # Save the data to CSV
    eth_data.to_csv('eth_data.csv', index=False)
    print("Data saved to eth_data.csv")

    # Live update the latest price every minute
    while True:
        latest_price = get_latest_eth_price(API_KEY)
        if latest_price is not None:
            latest_timestamp = pd.Timestamp.now()
            
            print(f"Latest ETH price at {latest_timestamp}: ${latest_price}")
            
            # Create a new DataFrame for the new row
            new_row = pd.DataFrame([{
                'Unix Timestamp': int(latest_timestamp.timestamp() * 1000),
                'Date': latest_timestamp,
                'Symbol': 'ETHUSD',
                'Open': latest_price,
                'High': latest_price,
                'Low': latest_price,
                'Close': latest_price,
                'Volume': None  # Volume data not available for live price updates
            }])
            
            # Concatenate the new row to the existing DataFrame
            eth_data = pd.concat([eth_data, new_row], ignore_index=True)
            
            # Save the updated data to CSV
            eth_data.to_csv('eth_data.csv', index=False)
        
        # Wait for a minute before fetching the next update
        time.sleep(60)
else:
    print("Failed to fetch initial data.")
