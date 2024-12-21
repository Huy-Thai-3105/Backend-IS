import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import seaborn as sns
sns.set_style('whitegrid')

from datetime import datetime
from vnstock3 import Vnstock

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

## GLOBAL VARIABLES ##
GLOBAL_STOCK     = Vnstock().stock(source='VCI')
DAY_INTERVAL_LEN = 60

## FUNCTION DEFINE ##
def LSTM_model(input_shape, stock_name):
    model = tf.keras.models.Sequential()
    model.name=f"{stock_name}_LSTM_model"
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.LSTM(units = 128, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units = 64, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units =25))
    model.add(tf.keras.layers.Dense(units =1))

    # model.summary()
    return model

def train_stock_model(stock_name):
    df = GLOBAL_STOCK.quote.history(symbol=stock_name, start='2018-01-01', end=datetime.now().strftime('%Y-%m-%d'), interval='1D')
    dataset = df.filter(['close']).values; # Now, use only the 'close' stock value
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(DAY_INTERVAL_LEN, len(train_data)):
        x_train.append(train_data[i - DAY_INTERVAL_LEN:i, 0])
        y_train.append(train_data[i, 0])
            
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = LSTM_model(input_shape=(x_train.shape[1], 1), stock_name=stock_name)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=64, epochs=10)

    model.save(f"MODEL/{stock_name}_LSTM_model.keras")

def predict_stock_price(stock_name, number_of_days):
    if not os.path.isfile(f"MODEL/{stock_name}_LSTM_model.keras"):
        print("Model not exists!!!!")
        return

    model = tf.keras.models.load_model(f"MODEL/{stock_name}_LSTM_model.keras")
    model.summary()

    df = GLOBAL_STOCK.quote.history(symbol=stock_name, start='2018-01-01', end=datetime.now().strftime('%Y-%m-%d'), interval='1D')
    dataset = df.filter(['close']).values; # Now, use only the 'close' stock value
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    predict_result = []
    input_data = [scaled_data[len(scaled_data) - DAY_INTERVAL_LEN:].flatten()]
    for _ in range(number_of_days):
        x_input = np.array(input_data)
        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1 ))

        # Get the models predicted price values 
        y_predict = model.predict(x_input)
        predict_value = scaler.inverse_transform(y_predict)[0][0]

        input_data = [np.append(input_data[0][1:], y_predict[0][0])]
        predict_result.append(float(predict_value))

    # print(predict_result)
    return predict_result

def show_result(stock_name, predict_data):
    df = GLOBAL_STOCK.quote.history(symbol=stock_name, start='2018-01-01', end=datetime.now().strftime('%Y-%m-%d'), interval='1D')
    current_data = df.filter(['close']).values.flatten().tolist(); # Now, use only the 'close' stock value
    total_df   = pd.DataFrame(current_data + predict_data, columns = ['value'])
    current_df = total_df[:len(current_data)]
    predict_df = total_df[len(current_data) - 1:]

    # Visualize the data
    plt.figure(figsize=(16,6))
    plt.title(f'{stock_name}_predict_model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(current_df)
    plt.plot(predict_df)
    plt.legend(['Actual', 'Predictions'], loc='lower right')
    plt.show()

if __name__ == "__main__":
    STOCK_COMPANY_LIST = ['FPT', 'ACB', 'BID', 'VIC']

    for stock_symbol in STOCK_COMPANY_LIST:
        train_stock_model(stock_name=stock_symbol); # This function is used train and save LSTM model
    #     predict_data = predict_stock_price(stock_name=stock_symbol, number_of_days=30); # This function is used to load and predict data

    #     # print(predict_data)
    #     show_result(stock_symbol, predict_data); # This function is used to display the result only