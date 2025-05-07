import os
import pandas as pd
from src.utils import tickers, fetch_stock_data, save_stock_data, load_stock_data, start_date, data_path
from datetime import datetime, timedelta

def main():
    # parse data from data directory
    for ticker in tickers:
        # check if data exists
        if not os.path.exists(os.path.join(data_path, f'{ticker}.csv')):
            # if not, download and save data
            print(f'{ticker} data not found. Downloading data...')
            df, full_name = fetch_stock_data(ticker, start_date)
            save_stock_data(df, ticker)
        else:
            # if data exists, load and check for any new data
            df_before = load_stock_data(ticker)
            last_date = df_before['ds'].max()
            next_date = last_date + timedelta(days=1)
            next_date_str = next_date.strftime('%Y-%m-%d')

            # download and add new data
            df, full_name = fetch_stock_data(ticker, next_date_str)
            if not df.empty:
                df = pd.concat([df_before, df])
                save_stock_data(df, ticker)
            else:
                print(f'No new data found for {ticker}')
        
    print('Data updated successfully!')
        

if __name__ == '__main__':
    main()