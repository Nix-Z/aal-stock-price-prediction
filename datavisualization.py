import plotly.express as px
import plotly.graph_objects as go
from data_preprocess import preprocess_data

def visualize_data():
    data = preprocess_data()

    # Time-Series Visualization
    fig_1 = px.line(data, x=data['date'], y=data['close'],
                    labels={
                        'date': 'Date',
                        'close': 'Closing Price (USD)'
                    },
                    title="Daily Closing Price of American Airlines Group Stock")
    fig_1.update_layout(showlegend=False)
    fig_1.update_xaxes(showgrid=False)
    fig_1.update_yaxes(showgrid=False)
    fig_1.show()

    # Candlestick Chart
    fig_2 = go.Figure(data=[go.Candlestick(x=data['date'],
                                           open=data['open'],
                                           high=data['high'],
                                           low=data['low'],
                                           close=data['close'])])
    fig_2.update_layout(title='American Airlines Group Stock Price Candlestick Chart',
                        xaxis_title='Date',
                        yaxis_title='Stock Price (USD)')
    fig_2.update_xaxes(showgrid=False)
    fig_2.update_yaxes(showgrid=False)
    fig_2.show()

    return data

visualize_data()
