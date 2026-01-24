from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('model/random_forest.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    history_df = pd.read_csv('data/daily_aggregated.csv')
    history_df['date'] = pd.to_datetime(history_df['date'])
    print("History data loaded.")
except Exception as e:
    print(f"Warning: Could not load history data: {e}")
    history_df = None

def get_historical_value(target_date_str):
    if history_df is None:
        return None
        
    target_date = pd.to_datetime(target_date_str)
    t_month = target_date.month
    t_day = target_date.day
    
    matches = history_df[
        (history_df['date'].dt.month == t_month) & 
        (history_df['date'].dt.day == t_day)
    ]
    
    if not matches.empty:
        return int(matches['ride_id_count'].mean())
    else:
        return None


def get_weather_forecast(date_str): #Get weather forecast for a given date form weather API

    lat = 40.7128
    lon = -74.0060

    # API URL (Daily forecast)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum,windspeed_10m_max&timezone=America%2FNew_York&start_date={date_str}&end_date={date_str}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        daily = data.get('daily', {})
        
        return {
            'temp_max': daily['temperature_2m_max'][0],   # °C
            'percip_total': daily['precipitation_sum'][0], # mm
            'wind_avg': daily['windspeed_10m_max'][0]      # km/h 
        }
    else:
        return None


def prepare_features(date_str, weather_data):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    
    month = date_obj.month
    day_of_week = date_obj.weekday() + 1 # 1=Mon, 7=Sun 
    
    month_angle = (month - 1) * 2 * np.pi / 12
    day_angle = (day_of_week) * 2 * np.pi / 7 
    
    features = {
        'month_sin': np.sin(month_angle),
        'month_cos': np.cos(month_angle),
        'day_of_week_sin': np.sin(day_angle),
        'day_of_week_cos': np.cos(day_angle),
        'wind_avg': weather_data['wind_avg'],
        'percip_total': weather_data['percip_total'],
        'temp_max': weather_data['temp_max']
    }
    
    cols_order = ['month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 'wind_avg', 'percip_total', 'temp_max']
    return pd.DataFrame([features])[cols_order]

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.json
    date_str = data.get('date')
    
    weather = get_weather_forecast(date_str)
    if not weather:
        return jsonify({'error': 'Could not fetch weather data'}), 400
        
    input_df = prepare_features(date_str, weather)
    
    prediction = model.predict(input_df)[0]
    
    historical_val = get_historical_value(date_str)
    
    return jsonify({
        'date': date_str,
        'prediction': int(prediction),
        'historical_value': historical_val,
        'weather': weather
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)