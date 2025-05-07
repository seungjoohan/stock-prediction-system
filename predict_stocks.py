import yfinance as yf
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from models.prophet import train_prophet_model, forecast_prophet
from models.arima import train_arima_model, forecast_arima
from src.utils import load_stock_data, tickers

# streamlit app
def main():
    st.title('Stock Price Prediction')
    
    # user input
    ticker = st.selectbox('Select a stock ticker', tickers)
    start_date = st.date_input('Start date', value=datetime.now() - timedelta(days=365))
    forecast_days = st.number_input('Forecast days', min_value=1, max_value=90, value=30)
    model_type = st.selectbox('Select model', ['Prophet', 'ARIMA'])
    
    # make prediction
    if st.button('Predict'):
        # fetch stock data
        with st.spinner('Fetching data...'):
            df = load_stock_data(ticker)
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
                forecast = forecast_prophet(st.session_state.model, forecast_days)
            elif model_type == "ARIMA":
                forecast = forecast_arima(st.session_state.model, df, forecast_days)
            
            st.session_state.forecast = forecast
            msg.success('Prediction generated successfully!')

        # Display results
        st.subheader('Stock Price Forecast')
        fig = px.line(df, x='ds', y='y', title=f'{ticker} Stock Price Forecast using {model_type}')
        fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast')
        if model_type == "Prophet":
            fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash'))
            fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash'))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price')
        fig.update_xaxes(range=[start_date, datetime.now() + timedelta(days=forecast_days)])
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()