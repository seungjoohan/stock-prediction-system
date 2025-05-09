from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from datetime import datetime, timedelta
import pandas as pd

model_artifacts_path = 'artifacts/'

def train_arima_model(df):
    model = ARIMA(df['y'], order=(5,0,1))
    model_arima = model.fit()
    return model_arima
    
def forecast_arima(model, df, periods):
    result = model.forecast(steps=periods)
    forecast = pd.DataFrame({'idx': result.index, 'yhat': result.values})
    last_date = df['ds'].max()
    next_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    forecast['ds'] = forecast['idx'].apply(lambda x: datetime.strptime(next_date, '%Y-%m-%d') + timedelta(days=x - len(df)))
    forecast['ds'] = pd.to_datetime(forecast['ds']).dt.tz_localize(None)
    return forecast

def save_arima_model(model):
    model.save(model_artifacts_path + 'arima_model.pkl')

def load_arima_model():
    model = ARIMAResults.load(model_artifacts_path + 'arima_model.pkl')
    return model