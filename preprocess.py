import pandas as pd
from ta.momentum import RSIIndicator
import pandas as pd
import ta
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import numpy as np
def preprocessing(dataset,lookback_factor):
    lookback_factor=1
    df = dataset
    df['Date'] = pd.to_datetime(df['Date'])
    # Assuming 'Date' column is already in datetime format

    # Define the threshold date for usability
    # Filter the DataFrame based on the threshold date and make a copy
    df_calculated = df

    # Initialize RSI Indicator
    indicator_rsi = RSIIndicator(close=df_calculated["Close"], window=6*int(lookback_factor))

    # Add RSI to DataFrame
    df_calculated['rsi'] = indicator_rsi.rsi()
    indicator_bb = BollingerBands(close=df_calculated["Close"], window=6*int(lookback_factor), window_dev=5*int(lookback_factor))

    # Add Bollinger Bands to DataFrame
    df_calculated['bb_hband'] = indicator_bb.bollinger_hband()
    df_calculated['bb_lband'] = indicator_bb.bollinger_lband()

    # Add Long Term Factors 
    df_calculated['ETH_50MA'] = df['Close'].rolling(window=10*int(lookback_factor)).mean()
    df_calculated['ETH_200MA'] = df['Close'].rolling(window=20*int(lookback_factor)).mean()

    # Display the DataFrame with RSI

    # Filter the DataFrame based on the threshold date and make a copy
    df_calculated = df_calculated
    df_final = df_calculated.copy()

    df = df_final
    # Define the threshold dates
    # Filter the DataFrame based on the threshold dates
    # Split the usable data into training and testing sets
    df_usable = df.iloc[40:-1]
    # Print the resulting DataFrames
    print("Data above usable threshold date:")
    return df_usable
    # Save df_usable to a CSV file
def create_sequences(data, look_back, look_ahead):
    X, y = [], []
    for i in range(look_back, len(data) - look_ahead):
        X.append(data[i-look_back:i])
        y.append(data[i + look_ahead, 3])  # Assuming the fourth column is the target (ETH_Close)
    return np.array(X), np.array(y)
    # Combine ETH prices and engineered features into one array