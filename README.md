# Ethereum Price Prediction with CNN-LSTM Model

## Introduction

Cryptocurrencies, like Ethereum (ETH), have gained significant attention in recent years due to their volatility and potential for high returns. Accurate prediction of cryptocurrency prices can be highly beneficial for investors and traders. This project aims to develop a predictive model for Ethereum prices using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The model leverages historical price data and technical indicators to forecast future prices.

## Project Overview

### Objectives

1. **Preprocess historical Ethereum price data**: Prepare the dataset for modeling by cleaning, transforming, and engineering features.
2. **Develop a hybrid CNN-LSTM model**: Use a combination of CNN and LSTM layers to capture both spatial and temporal patterns in the data.
3. **Evaluate model performance**: Assess the accuracy of the model's predictions using various metrics and visualizations.
4. **Combine predictions using linear regression**: Aggregate predictions from multiple models with different lookback periods to improve overall accuracy.

### Tools and Libraries

- **Python**: Main programming language used for the project.
- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical computations.
- **TA-Lib**: Technical analysis library for feature engineering.
- **Scikit-Learn**: Machine learning library for preprocessing and linear regression.
- **Keras**: Deep learning library for building and training neural networks.
- **Matplotlib**: Visualization library for plotting results.

## Data Preprocessing

### Loading and Preparing Data

The first step involves loading historical Ethereum price data from a CSV file and preparing it for modeling. This includes handling missing values, converting date columns to datetime format, and sorting the data chronologically.

```python
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def preprocessing(dataset, lookback_factor):
    df = dataset
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # Initialize RSI Indicator
    indicator_rsi = RSIIndicator(close=df["Close"], window=6*int(lookback_factor))

    # Add RSI to DataFrame
    df['rsi'] = indicator_rsi.rsi()

    # Initialize Bollinger Bands
    indicator_bb = BollingerBands(close=df["Close"], window=6*int(lookback_factor), window_dev=5*int(lookback_factor))

    # Add Bollinger Bands to DataFrame
    df['bb_hband'] = indicator_bb.bollinger_hband()
    df['bb_lband'] = indicator_bb.bollinger_lband()

    # Add Moving Averages
    df['ETH_50MA'] = df['Close'].rolling(window=10*int(lookback_factor)).mean()
    df['ETH_200MA'] = df['Close'].rolling(window=20*int(lookback_factor)).mean()

    # Filter usable data
    df_usable = df.iloc[40:-1]

    return df_usable
```

### Feature Engineering

Technical indicators such as the Relative Strength Index (RSI) and Bollinger Bands are used to enrich the dataset with additional features that can help the model understand market conditions.

## Model Development

### Creating Sequences for Modeling

To prepare the data for the CNN-LSTM model, we need to create sequences of past observations to predict future values. This involves defining a lookback period and a lookahead period.

```python
def create_sequences(data, look_back, look_ahead):
    X, y = [], []
    for i in range(look_back, len(data) - look_ahead):
        X.append(data[i - look_back:i])
        y.append(data[i + look_ahead, 3])  # Assuming the fourth column is the target (ETH_Close)
    return np.array(X), np.array(y)
```

### Building the CNN-LSTM Model

The model architecture combines convolutional layers for spatial feature extraction and LSTM layers for capturing temporal dependencies.

```python
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout

def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(75, activation='relu', return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
```

### Training the Model

The model is trained using historical data with early stopping and learning rate reduction to prevent overfitting.

```python
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(X_train, y_train, X_val, y_val, batch_size):
    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=10, min_lr=0.003)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=80, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])
    return model
```

## Model Evaluation

### Making Predictions

After training the model, predictions are made on the test set and compared to the actual values to evaluate performance.

```python
def evaluate_model(model, X_test, y_test, scaler_y):
    predictions = model.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions)
    y_test_inv = scaler_y.inverse_transform(y_test)

    plt.figure(figsize=(14, 7))
    plt.plot(y_test_inv, label='Actual ETH Price', color='black')
    plt.plot(predictions, label='Predicted ETH Price', linestyle='dashed')
    plt.legend()
    plt.title('Actual vs. Predicted ETH Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
```

### Combining Predictions

To enhance prediction accuracy, outputs from multiple models with different lookback factors are combined using linear regression.

```python
from sklearn.linear_model import LinearRegression

def combine_predictions(train_preds, test_preds, y_train):
    model_lr = LinearRegression()
    model_lr.fit(train_preds, y_train)
    combined_predictions = model_lr.predict(test_preds)
    return combined_predictions
```

### Measuring Accuracy

The directional accuracy of the predictions is measured by comparing the predicted price changes to the actual price changes.

```python
def calculate_directional_accuracy(actual, predicted, look_ahead):
    actual_changes = np.where(actual > np.roll(actual, -look_ahead), 1, 0)
    predicted_changes = np.where(predicted > np.roll(predicted, -look_ahead), 1, 0)
    difference = actual_changes.flatten() - predicted_changes.flatten()
    num_zeros = np.count_nonzero(difference == 0)
    percentage_zeros = (num_zeros / len(difference)) * 100
    print(f"Number of matching directions: {num_zeros}")
    print(f"Percentage of matching directions: {percentage_zeros:.2f}%")
```

## Conclusion

This project demonstrates a robust approach to predicting Ethereum prices using a hybrid CNN-LSTM model. By leveraging historical price data and technical indicators, the model captures both spatial and temporal patterns in the data. The combination of multiple models with different lookback periods further enhances the prediction accuracy. This approach can be extended to other cryptocurrencies and financial assets for similar predictive modeling tasks.

### Future Work

- **Hyperparameter Tuning**: Experiment with different hyperparameters to further optimize the model.
- **Incorporate Additional Features**: Use other technical indicators or fundamental data to enrich the feature set.
- **Real-Time Predictions**: Implement the model in a real-time trading system to provide live predictions.

This project provides a solid foundation for further exploration and development in the field of cryptocurrency price prediction.