import pandas as pd
import numpy as np
import warnings
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as py
import plotly.express as px
from sklearn.metrics import mean_squared_error
from datavisualization import visualize_data

warnings.simplefilter("ignore", category=FutureWarning)

def prophet_model():
    data = visualize_data()

    # Select data features
    data.drop(['open', 'high', 'low', 'volume'], axis=1, inplace=True)
    data = data.rename(columns = {'date': 'ds', 'close': 'y'})

    # Split data into train and test datasets
    n = len(data)
    train_size = int(n * 0.8)
    test_size = n - train_size
    train, test = data[0 : train_size], data[:] # Change test to full length of data
    test.drop(['y'], axis=1, inplace=True)

    # Build model
    model = Prophet(interval_width=0.95)
    model.fit(train)
    forecast = model.predict(test)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

    fig = model.plot(forecast)
    fig.show()

    # Evaluate model performance using Root Mean Squared Error (RMSE)
    testScore = np.sqrt(mean_squared_error(data['y'], forecast['yhat']))
    print('Forecast Score: %.2f RMSE' % (testScore))

    # Visualization of predicted vs actual stock price
    graphdata = pd.DataFrame(
        {
            'Date': data['ds'],
            'Actual': data['y'],
            'Forecast': forecast['yhat'],
        },
    )

    fig_1 = px.line(graphdata, x='Date', y=['Actual', 'Forecast'], labels={'variable': 'Key'},
                    title="LSTM Model Predicting AAL Closing Stock Prices"
                    )
    fig_1.update_layout(xaxis_title='Date', yaxis_title='Closing Price (USD)')
    fig_1.update_xaxes(showgrid=False)
    fig_1.update_yaxes(showgrid=False)
    fig_1.show()

    data.to_csv('AAL_cleansed_data.csv', index=False)

    return data

prophet_model()
