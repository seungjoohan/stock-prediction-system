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
 - Refresh data every day 
 - Retrain models (ARIMA / Prophet)

## TODO
 - Model auto-update
 - Predict when client requests come in
 - Model performance monitoring system
    * metrics already saved in `artifacts` directory
    * implement hyperparamters tuning when performance insufficient
