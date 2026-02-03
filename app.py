import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Citi Bike Demand Predictor",
    layout="centered",
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('resources/random_forest.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load historical data
@st.cache_data
def load_history():
    try:
        history_df = pd.read_csv('resources/daily_aggregated.csv')
        history_df['date'] = pd.to_datetime(history_df['date'])
        return history_df
    except Exception as e:
        st.warning(f"Could not load history data: {e}")
        return None

model = load_model()
history_df = load_history()

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


def get_weather_forecast(date_str):
    """Get weather forecast for a given date from weather API"""
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

# Streamlit UI
st.title("Citi Bike Demand Predictor")

# Sidebar for input
st.sidebar.header("Settings")

# Date input
min_date = datetime.now().date()
max_date = min_date + timedelta(days=14)  # Forecast available for next 14 days
selected_date = st.sidebar.date_input(
    "Select Date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

# Convert to string format
date_str = selected_date.strftime("%Y-%m-%d")

# Prediction button
if st.sidebar.button("Predict", type="primary"):
    if not model:
        st.error("Model not loaded. Please check the model file.")
    else:
        with st.spinner("Fetching weather data and making prediction..."):
            # Get weather forecast
            weather = get_weather_forecast(date_str)
            
            if not weather:
                st.error("Could not fetch weather data. Please try again later.")
            else:
                # Prepare features and make prediction
                input_df = prepare_features(date_str, weather)
                prediction = model.predict(input_df)[0]
                
                # Get historical value
                historical_val = get_historical_value(date_str)
                
                st.success("Prediction completed!")
                
                col1, col2 = st.columns(2)
                with col1:

                    st.metric(
                        label="Date",
                        value=selected_date.strftime("%B %d, %Y")
                    )

                with col2:
                    st.metric(
                        label="Day of Week",
                        value=selected_date.strftime("%A")
                    )

                col1, col2 = st.columns(2)

                with col1:
                    if historical_val:
                        st.metric(
                            label="Predicted Rides",
                            value=f"{int(prediction):,}",
                            delta=f"{int(prediction - historical_val):,} than in 2023"
                        )
                    else:
                        st.metric(
                            label="Predicted Rides",
                            value=f"{int(prediction):,}"
                        )
                
                with col2:
                    pass
                
                # Weather information
                st.subheader("Weather Forecast")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Max Temperature",
                        value=f"{weather['temp_max']:.1f}°C"
                    )
                
                with col2:
                    st.metric(
                        label="Precipitation",
                        value=f"{weather['percip_total']:.1f} mm"
                    )
                
                with col3:
                    st.metric(
                        label="Wind Speed",
                        value=f"{weather['wind_avg']:.1f} km/h"
                    )


# Information section
with st.expander("About this app"):
    st.markdown("""
    This application predicts the number of bike rides in New York City using:
    - **Machine Learning Model**: Random Forest trained on historical bike ride data from 2023
    - **Weather Data**: Real-time weather forecasts from Open-Meteo API
    
    The model considers:
    - Month and day of week (cyclical encoding)
    - Maximum temperature
    - Precipitation
    - Wind speed
                
    The predictions can help in planning and resource allocation for bike-sharing services.
    """)

# Footer
st.caption("Data source: NYC Citi Bike | Weather: Open-Meteo API")