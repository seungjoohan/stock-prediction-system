# Stock Prediction System

This is a personal project to build the system that systematically parse stock data, train the model, and deploy the app using Streamlit. 

As I am focusing on learning ML system architecture, I will focus on model production than training very accurate model.

Let's use TSLA, AAPL, GOOG.

## Run Streamlit app

```bash
streamlit run predict_stocks.py
```

## Current implementation

 - Deploying stock prediction app using Streamlit.
 - Train very simple chosen (Prophet/ARIMA) model on the fly
 - Generate prediction plots
 - Automatically download data files at 00:00 UTC every day or when manually activate 'Update Stock Data' action.

## TODO

 - Add model validation
 - Train & save default model - prophet / arima / vaniall NN
 - Model auto-update with new data
 - Predict when client requests come in