import pandas as pd
from data_analysis import analyze_data

def preprocess_data():
    data = analyze_data()
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")
    data.drop(['Name'], axis=1, inplace=True)
    print(data.head())

    return data

preprocess_data()
