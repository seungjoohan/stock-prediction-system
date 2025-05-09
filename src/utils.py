import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

tickers = ['TSLA', 'AAPL', 'GOOG']
data_path = 'data'
start_date = '2024-01-01'
artifacts_path = 'artifacts'

# Validation metrics
def mape(y_true, y_pred):
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)

def rmse(y_true, y_pred):
    return round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 2)

def mae(y_true, y_pred):
    return round(np.mean(np.abs(y_true - y_pred)), 2)

def save_metrics(metrics, ticker):
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(artifacts_path, f'{ticker}_metrics.csv'), index=False)

# fetch stock data
def fetch_stock_data(ticker, start_date):
    stock = yf.Ticker(ticker)
    full_name = stock.info.get('longName', 'N/A')
    df = stock.history(start=start_date)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})  # Prophet format
    return df, full_name

def save_stock_data(df, ticker):
    df.to_csv(os.path.join(data_path, f'{ticker}.csv'), index=False)
    print(f'{ticker} data saved')

def load_stock_data(ticker):
    df = pd.read_csv(os.path.join(data_path, f'{ticker}.csv'), parse_dates=['ds'])
    return df

if __name__ == '__main__':
    from prophet import Prophet
    df = load_stock_data('TSLA')
    model = Prophet()
    arima_metrics, prophet_metrics = walk_forward_validation(df)
    print(arima_metrics)
    print(prophet_metrics)
    # best_model = determine_best_model(arima_metrics, prophet_metrics)
    # print(best_model)