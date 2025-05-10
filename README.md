# Stock Prediction System

This is a personal project to build the system that systematically parse stock data, train the model, and deploy the app using Streamlit. 

As I am focusing on learning ML system architecture, I will focus on model production / maintenance than training very accurate model.

Let's use TSLA, AAPL, GOOG.

## Run Streamlit app

```bash
streamlit run predict_stocks.py
```

## Current implementation

 - Deploying stock prediction app using Streamlit.
 - Generate prediction plots
 - Refresh data every day through yfinance API
 - Retrain models (ARIMA / Prophet) & save walk forward cross validation results
 - When asked to predict, the system loads the model instead of training on the fly.

## TODO
 - Predict when client requests come in
 - Model performance monitoring system
    * metrics saved in `artifacts` directory
    * implement hyperparamters tuning when performance insufficient
