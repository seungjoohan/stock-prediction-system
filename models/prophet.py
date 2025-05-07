from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

model_artifacts_path = 'artifacts/'

def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model

def forecast_prophet(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

def save_prophet_model(model):
    model_json = model_to_json(model)
    with open(model_artifacts_path + 'prophet_model.json', 'wb') as f:
        f.write(model_json)

def load_prophet_model():
    with open(model_artifacts_path + 'prophet_model.json', 'rb') as f:
        model = model_from_json(f.read())
    return model