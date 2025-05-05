import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import prophet
import statsmodels.tsa.arima.model as arima
import streamlit as st
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# fetch stock data
def fetch_stock_data(stock, start_date, end_date):
    df = stock.history(start=start_date, end=end_date)
    df = df[['Close']].reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})  # Prophet format
    return df

# train prophet model
def train_prophet_model(df):
    model = prophet.Prophet()
    model.fit(df)
    return model

# train arima model
def train_arima_model(df):
    model = arima.ARIMA(df['y'], order=(5,0,1))
    model_arima = model.fit()
    return model_arima

# prophet forecast
def prophet_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods)
    print(future)
    forecast = model.predict(future)
    print(forecast)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

# arima forecast
def arima_forecast(model, periods):
    forecast = model.forecast(steps=periods)
    return forecast

# streamlit app
def main():
    st.title('Stock Price Prediction')
    
    # user input
    ticker = st.text_input('Enter a stock ticker (e.g. TSLA)', value='TSLA')
    start_date = st.date_input('Start date', value=datetime.now() - timedelta(days=365))
    forecast_days = st.number_input('Forecast days', min_value=1, max_value=90, value=30)
    model_type = st.selectbox('Select model', ['Prophet', 'ARIMA'])

    # default end date to today
    end_date = datetime.now()

    # fetch stock data
    stock = yf.Ticker(ticker)
    full_name = stock.info.get('longName', 'N/A')
    
    # make prediction
    if st.button('Predict'):
        with st.spinner('Fetching data...'):
            df = fetch_stock_data(stock, start_date, end_date)
            if df.empty:
                st.error('No data found for the given ticker and date range. Check the ticker and date range.')
                return

            st.session_state.data = df
            st.session_state.ticker = ticker
            msg = st.success('Data fetched successfully!')
        
        with st.spinner('Training model...'):
            if model_type == "Prophet":
                st.session_state.model = train_prophet_model(df)
            else:
                st.session_state.model = train_arima_model(df)

            st.session_state.model_type = model_type
            msg.success('Model trained successfully!')

        with st.spinner('Generating forecast...'):
            if model_type == "Prophet":
                forecast = prophet_forecast(st.session_state.model, forecast_days)
            elif model_type == "ARIMA":
                result = arima_forecast(st.session_state.model, forecast_days)
                forecast = pd.DataFrame({'idx': result.index, 'yhat': result.values})
                forecast['ds'] = forecast['idx'].apply(lambda x: datetime.now() + timedelta(days=x - len(st.session_state.data)))
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.tz_localize(None)
            
            st.session_state.forecast = forecast
            msg.success('Prediction generated successfully!')

        # Display results
        st.subheader('Stock Price Forecast')
        fig = px.line(df, x='ds', y='y', title=f'{full_name} Stock Price Forecast using {model_type}')
        fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast')
        if model_type == "Prophet":
            fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash'))
            fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash'))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price')
        fig.update_xaxes(range=[start_date, datetime.now() + timedelta(days=forecast_days)])
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()