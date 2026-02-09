import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

daily = pd.read_csv('resources/daily_aggregated.csv')
features = ['month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 'wind_avg', 'percip_total', 'temp_max']
target = 'ride_id_count'
X = daily[features]
y = daily[target]


loaded_model = joblib.load('resources/random_forest.pkl')

y_pred = loaded_model.predict(X)

r2 = r2_score(y_pred, y)

print ("Sample Predictions:", y_pred[:5])

print("R2 Score on training data:", r2)

