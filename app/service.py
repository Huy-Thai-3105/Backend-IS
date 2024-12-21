import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime
import numpy as np
import pandas as pd
from vnstock3 import Vnstock

# Khởi tạo biến toàn cục
GLOBAL_STOCK = Vnstock().stock(source='VCI')
DAY_INTERVAL_LEN = 60

def predict_stock_price(stock_name, number_of_days=30):
    if not os.path.isfile(f"MODEL/{stock_name}_LSTM_model.keras"):
        return None

    model = tf.keras.models.load_model(f"MODEL/{stock_name}_LSTM_model.keras")
    
    # Lấy dữ liệu lịch sử
    df = GLOBAL_STOCK.quote.history(symbol=stock_name, start='2018-01-01', end=datetime.now().strftime('%Y-%m-%d'), interval='1D')
    # Lấy 30 ngày gần nhất cho historical data
    historical_data = df.tail(30)['close'].tolist()
    
    # Chuẩn bị dữ liệu cho prediction
    dataset = df.filter(['close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Dự đoán cho 30 ngày trước
    past_predictions = []
    input_data  = [scaled_data[len(scaled_data) - DAY_INTERVAL_LEN - 30:len(scaled_data) - 30].flatten()]
    append_data = [scaled_data[len(scaled_data) - 30:].flatten()]
    print(append_data)
    
    for idx in range(number_of_days):
        x_input = np.array(input_data)
        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1)) 

        y_predict = model.predict(x_input)
        predict_value = scaler.inverse_transform(y_predict)[0][0]

        input_data = [np.append(input_data[0][1:], append_data[0][idx])]
        past_predictions.append(float(predict_value))
        
        

    # Dự đoán cho 30 ngày tiếp theo
    future_predictions = []
    input_data = [scaled_data[len(scaled_data) - DAY_INTERVAL_LEN:].flatten()]
    
    for idx in range(number_of_days):
        x_input = np.array(input_data)
        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

        y_predict = model.predict(x_input)
        predict_value = scaler.inverse_transform(y_predict)[0][0]

        input_data = [np.append(input_data[0][1:], append_data[0][idx])]
        future_predictions.append(float(predict_value))

    return {
        'historical_data': historical_data,    
        'past_predictions': past_predictions,    
        'future_predictions': future_predictions 
    }

def get_prediction(stock_code, number_of_days=30):
    result = predict_stock_price(stock_name=stock_code, number_of_days=number_of_days)
    if result is None:
        return None
        
    return {
        'historical': [
            {
                'time': (datetime.now() - pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
                'value': value,
                'predicted': pred_value
            } for i, (value, pred_value) in enumerate(zip(reversed(result['historical_data']), 
                                                        reversed(result['past_predictions'])), 1)
        ],
        'predictions': [
            {
                'time': (datetime.now() + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
                'value': value
            } for i, value in enumerate(result['future_predictions'], 1)
        ]
    }

def get_stock_data(stock_code, start, end, interval='1D'):
    """
    Lấy dữ liệu cổ phiếu với khoảng thời gian tùy chọn
    """
    try:
        # Kiểm tra nếu là intraday thì thông báo không hỗ trợ
        if interval in ['1m', '5m', '15m', '30m', '1H']:
            print("Intraday data is not supported")
            return {
                "error": "Dữ liệu theo giờ và phút hiện không được hỗ trợ. Vui lòng sử dụng interval 1D, 1W hoặc 1M."
            }
            
        # Xử lý cho dữ liệu daily, weekly và monthly
        end = datetime.now()
        if interval == '1D':
            # Lấy 30 ngày gần nhất
            start = (end - pd.Timedelta(days=30))
        elif interval == '1W':
            # Lấy 24 tuần gần nhất
            start = (end - pd.Timedelta(weeks=24))
        elif interval == '1M':
            # Lấy 12 tháng gần nhất
            start = (end - pd.Timedelta(days=365))

        # Convert to string format for API call
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')


        print(f"Fetching data for {stock_code} from {start_str} to {end_str} with interval {interval}")
        
        df = GLOBAL_STOCK.quote.history(
            symbol=stock_code,
            start=start_str,
            end=end_str,
            interval=interval
        )
        
        if df.empty:
            print("No data returned")
            return None
            
        print(f"Retrieved {len(df)} records")
        
        # Chuyển đổi DataFrame thành dictionary với các trường đã định dạng
        result = []
        for _, row in df.iterrows():
            data_point = {
                'time': pd.to_datetime(row['time']).strftime('%Y-%m-%d'),  # Sử dụng cột 'time'
                'open': float(row['open']),
                'close': float(row['close']),
                'high': float(row['high']),
                'low': float(row['low']),
                'volume': float(row['volume'])
            }
            result.append(data_point)
        
        return result
        
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def calculate_technical_indicators(stock_code):
    try:
        # Lấy dữ liệu nhiều hơn để tính toán chính xác
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = GLOBAL_STOCK.quote.history(
            symbol=stock_code,
            start=(datetime.now() - pd.Timedelta(days=28)).strftime('%Y-%m-%d'),  # Lấy 28 ngày
            end=end_date,
            interval='1D'
        )
        
        if df.empty:
            print("No data returned from API")
            return None
            
        # Đảm bảo index là datetime
        df.index = pd.to_datetime(df.index)
            
        # Tính SMA và EMA
        df['sma14'] = df['close'].rolling(window=14, min_periods=1).mean()
        df['ema14'] = df['close'].ewm(span=14, adjust=False).mean()
        
        # Tính RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi14'] = 100 - (100/(1 + rs))

        # Chỉ lấy 14 ngày gần nhất
        df = df.tail(14)

        result = []
        for index, row in df.iterrows():
            data_point = {
                'timestamp': index.strftime('%Y-%m-%d'),
                'symbol': stock_code,
                'open': float(row['open']),
                'close': float(row['close']),
                'max': float(row['high']),
                'min': float(row['low']),
                'volume': float(row['volume']),
                'sma14': float(row['sma14']),
                'ema14': float(row['ema14']),
                'rsi14': float(row['rsi14'])
            }
            result.append(data_point)

        return result

    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def calculate_sma(df, period=20):
    """Tính Simple Moving Average"""
    sma = df['close'].rolling(window=period).mean()
    return sma.dropna().tolist()

def calculate_ema(df, period=20):
    """Tính Exponential Moving Average"""
    ema = df['close'].ewm(span=period, adjust=False).mean()
    return ema.dropna().tolist()

def calculate_rsi(df, period=14):
    """Tính Relative Strength Index"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.dropna().tolist()

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Tính MACD (Moving Average Convergence Divergence)"""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return {
        'macd': macd.dropna().tolist(),
        'signal': signal_line.dropna().tolist(),
        'histogram': histogram.dropna().tolist()
    }

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Tính Bollinger Bands"""
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return {
        'middle': sma.dropna().tolist(),
        'upper': upper_band.dropna().tolist(),
        'lower': lower_band.dropna().tolist()
    }