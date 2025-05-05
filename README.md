# Stock Prediction System

    This is a personal project to build the system that systematically parse stock data, train the model, and deploy the app using Streamlit. 

    I am focusing on learning ML system architecture, I will not focus on training the accurate model.

    ## Predicting stock price 
    ```bash
    streamlit run predict_stocks.py
    ```

    ## Current implementation
        Deploying stock prediction app using Streamlit.
        Train very simple chosen (Prophet/ARIMA) model on the fly
        Generate prediction plots

    ## TODO
        Add model validation
        Train & save default model - prophet / arima / vaniall NN 
        Streaming-like data updates: not parsing data on the fly, but updating data every week / day
        Model auto-update with new data
        Predict when client requests come in