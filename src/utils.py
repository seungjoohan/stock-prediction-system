import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

tickers = ['TSLA', 'AAPL', 'GOOG']
data_path = 'data'

# Mean Absolute Percentage Error to evaluate models
def mape(y_true, y_pred):
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)

# fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    full_name = stock.info.get('longName', 'N/A')
    df = stock.history(start=start_date, end=end_date)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})  # Prophet format
    return df, full_name

# def refresh_data():
#     for ticker in tickers:
#         df, full_name = fetch_stock_data(ticker, start_date, end_date)
#         df.to_csv(f'data/{ticker}.csv', index=False)
#         print(f'{ticker} data refreshed')

