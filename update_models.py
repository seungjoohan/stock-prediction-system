from models.prophet import train_prophet_model, forecast_prophet, save_prophet_model
from models.arima import train_arima_model, forecast_arima, save_arima_model
from src.utils import mape, rmse, mae, tickers, data_path, load_stock_data, save_metrics
import pandas as pd
import os

artifacts_path = 'artifacts'

def walk_forward_validation(df, n_folds=5):
    metrics = {'arima_mape': [], 'arima_rmse': [], 'arima_mae': [], 
               'prophet_mape': [], 'prophet_rmse': [], 'prophet_mae': []}
    base_train_len = int(0.5 * len(df))

    for fold in range(n_folds):
        # split data into train and test
        step_size = int((len(df) - base_train_len) / (n_folds+1) * (fold + 1))
        train_data = df.iloc[:base_train_len + step_size]
        test_data = df.iloc[base_train_len + step_size: base_train_len + 2 * step_size]

        # train and validate model
        prophet_model = train_prophet_model(train_data)
        prophet_forecast = forecast_prophet(prophet_model, len(test_data))
        arima_model = train_arima_model(train_data)
        arima_forecast = forecast_arima(arima_model, train_data, len(test_data))
        
        # calculate metrics
        metrics['prophet_mape'].append(mape(test_data['y'], prophet_forecast['yhat']))
        metrics['prophet_rmse'].append(rmse(test_data['y'], prophet_forecast['yhat']))
        metrics['prophet_mae'].append(mae(test_data['y'], prophet_forecast['yhat']))
        metrics['arima_mape'].append(mape(test_data['y'], arima_forecast['yhat'].values))
        metrics['arima_rmse'].append(rmse(test_data['y'], arima_forecast['yhat'].values))
        metrics['arima_mae'].append(mae(test_data['y'], arima_forecast['yhat'].values))

    return metrics

def main():
    os.makedirs(artifacts_path, exist_ok=True)

    for ticker in tickers:
        df = load_stock_data(ticker)
        metrics = walk_forward_validation(df)

        # TODO: 
            # set baseline performance
            # check metrics
            # tune hyperparameters
            # redo walk forward validation
            # repeat until performance is acceptable

        # save metrics
        save_metrics(metrics, ticker)
        
        # train final model with all data and save
        prophet_model = train_prophet_model(df)
        arima_model = train_arima_model(df)
        save_prophet_model(prophet_model, ticker)
        save_arima_model(arima_model, ticker)

if __name__ == '__main__':
    main()