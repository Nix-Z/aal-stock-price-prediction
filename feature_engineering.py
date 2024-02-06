import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from datavisualization import visualize_data


def engineer_features():
    data = visualize_data()

    # Select data feature
    selected_data = data.filter(['close']).values
    print(selected_data)

    # Normalize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)
    
    # Split data into training and testing sets
    n = len(scaled_data)
    train_size = int(n * 0.8)
    test_size = n - train_size
    train, test = scaled_data[0 : train_size, : ], scaled_data[train_size : n, : ]
    print(len(train), len(test))

    # Convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        X_data, Y_data = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i : (i + look_back), 0]
            X_data.append(a)
            Y_data.append(dataset[i + look_back, 0])
        return np.array(X_data), np.array(Y_data)

    # Create train and test datasets for both X and Y
    look_back = 60
    train_X, train_Y = create_dataset(train, look_back)
    test_X, test_Y = create_dataset(test, look_back)

    # Reshape data
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape)

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(1, look_back)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_X, train_Y, epochs=20, batch_size=1)

    # Make predictions
    predict_train = model.predict(train_X)
    predict_test = model.predict(test_X)
    
    # Invert predictions
    predict_train = scaler.inverse_transform(predict_train)
    train_Y = scaler.inverse_transform([train_Y])
    predict_test = scaler.inverse_transform(predict_test)
    test_Y = scaler.inverse_transform([test_Y])
    print(predict_train.shape)

    # Evaluation model performance using Root Mean Squared Error (RMSE)
    trainScore = np.sqrt(mean_squared_error(train_Y[0], predict_train[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(test_Y[0], predict_test[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # Visualization of predicted vs actual stock prices
    predict_train = np.reshape(predict_train, len(predict_train))
    trainPredictPlot = np.full(len(selected_data), np.nan)
    trainPredictPlot[look_back:len(predict_train)+look_back] = predict_train

    predict_test = np.reshape(predict_test, len(predict_test))
    testPredictPlot = np.full(len(selected_data), np.nan)
    testPredictPlot[len(predict_train)+(look_back*2)+1:len(selected_data)-1] = predict_test

    graphdata = pd.DataFrame(
        {
            'Date': data['date'],
            'Actual': data['close'],
            'Train': trainPredictPlot,
            'Test': testPredictPlot,
        },
    )
    fig_1 = px.line(graphdata, x='Date', y=['Actual', 'Train', 'Test'],
                    labels={'variable': 'Key'},
                    title="LSTM Model Predicting AAL Closing Stock Prices"
                    )
    fig_1.update_layout(xaxis_title='Date', yaxis_title='Closing Price (USD)')
    fig_1.update_xaxes(showgrid=False)
    fig_1.update_yaxes(showgrid=False)
    fig_1.show()

    data.to_csv('AAL_cleansed_data.csv', index=False)
    
    return data

engineer_features()


    
    
