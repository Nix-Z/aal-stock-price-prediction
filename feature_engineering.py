import pandas as pd
import numpy as np
from datavisualization import visualize_data

def engineer_features():
    data = visualize_data()

    # Drop unnecessary data features
    data.drop(['open', 'high', 'low', 'volume'], axis=1, inplace=True)

    data.to_csv('AAL_cleansed_data.csv', index=False)

    return data

engineer_features()
