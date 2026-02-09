import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

from tools import get_historical_value, get_weather_forecast, prepare_features

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


# Streamlit UI
st.title("NYC: Citi Bike")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Analysis", "Prediction", "About"])

with tab1:
    st.subheader("Historical Data Analysis (2023)")
    
    if history_df is not None:
        max_rides_idx = history_df['ride_id_count'].idxmax()
        min_rides_idx = history_df['ride_id_count'].idxmin()
        max_rides_date = history_df.loc[max_rides_idx, 'date'].strftime('%B %d, %Y')
        min_rides_date = history_df.loc[min_rides_idx, 'date'].strftime('%B %d, %Y')
        
        # Overview metrics
        st.markdown("### Key Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Total Rides",
                value=f"{history_df['ride_id_count'].sum():,.0f}"
            )
        
        with col2:
            st.metric(
                label="Avg Daily Rides",
                value=f"{history_df['ride_id_count'].mean():,.0f}"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Most Rides in a Day",
                value=f"{history_df['ride_id_count'].max():,.0f}"
            )
            st.caption(f"{max_rides_date}")

        with col2:
            st.metric(
                label="Least Rides in a Day",
                value=f"{history_df['ride_id_count'].min():,.0f}"
            )
            st.caption(f"{min_rides_date}")

            


        
        st.markdown("---")
        
        # Daily rides and distance over time
        st.markdown("### Bike Ride Trends Over Time")
        fig1 = px.line(
            history_df,
            x='date',
            y='ride_id_count',
            title='Daily Rides in 2023',
            labels={'ride_id_count': 'Number of Rides', 'date': 'Date'}
        )
        fig1.update_traces(line_color='#0066cc')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig8 = px.line(
            history_df,
            x='date',
            y='distance_km_mean',
            title='Average Ride Distance Over Time',
            labels={'distance_km_mean': 'Avg Distance (km)', 'date': 'Date'}
        )
        fig8.update_traces(line_color='#ff6b35')
        st.plotly_chart(fig8, use_container_width=True)

        st.markdown("---")

        # Temperature vs Rides
        st.markdown("### Weather Impact on Bike Usage")
        fig2 = px.scatter(
            history_df,
            x='temp_max',
            y='ride_id_count',
            title='Maximum Temperature vs Daily Rides',
            labels={'temp_max': 'Max Temperature (°C)', 'ride_id_count': 'Number of Rides'},
            trendline='ols',
            color='temp_max',
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig2, use_container_width=True)

        col1, col2 = st.columns(2)
        
        with col1:
            fig6 = px.scatter(
                history_df,
                x='wind_avg',
                y='ride_id_count',
                title='Wind Speed vs Rides',
                labels={'wind_avg': 'Avg Wind Speed (km/h)', 'ride_id_count': 'Number of Rides'},
                trendline='ols',
                opacity=0.6
            )
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            fig7 = px.scatter(
                history_df,
                x='percip_total',
                y='ride_id_count',
                title='Precipitation vs Rides',
                labels={'percip_total': 'Precipitation (mm)', 'ride_id_count': 'Number of Rides'},
                trendline='ols',
                opacity=0.6
            )
            st.plotly_chart(fig7, use_container_width=True)
        
        st.markdown("---")

        # Monthly aggregation
        st.markdown("### Temporal analysis")
        history_df['month'] = pd.to_datetime(history_df['date']).dt.month
        history_df['month_name'] = pd.to_datetime(history_df['date']).dt.strftime('%B')
        monthly_data = history_df.groupby(['month', 'month_name'])['ride_id_count'].agg(['mean', 'sum']).reset_index()
        monthly_data = monthly_data.sort_values('month')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.bar(
                monthly_data,
                x='month_name',
                y='mean',
                title='Average Daily Rides by Month',
                labels={'mean': 'Avg Daily Rides', 'month_name': 'Month'},
                color='mean',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = px.bar(
                monthly_data,
                x='month_name',
                y='sum',
                title='Total Rides by Month',
                labels={'sum': 'Total Rides', 'month_name': 'Month'},
                color='sum',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig4, use_container_width=True)
        

        history_df['day_of_week'] = pd.to_datetime(history_df['date']).dt.dayofweek
        history_df['day_name'] = pd.to_datetime(history_df['date']).dt.strftime('%A')
        weekly_data = history_df.groupby(['day_of_week', 'day_name'])['ride_id_count'].mean().reset_index()
        weekly_data = weekly_data.sort_values('day_of_week')
        
        fig5 = px.bar(
            weekly_data,
            x='day_name',
            y='ride_id_count',
            title='Average Rides by Day of Week',
            labels={'ride_id_count': 'Avg Rides', 'day_name': 'Day'},
            color='ride_id_count',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig5, use_container_width=True)
        
        
    else:
        st.error("Historical data not available for analysis.")

with tab2:
    st.subheader("Citi Bike Demand Prediction")
    # Date input at the top of the page
    min_date = datetime.now().date()
    max_date = min_date + timedelta(days=14)  # Forecast available for next 14 days
    selected_date = st.date_input(
        "Pick a date for prediction",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    # Convert to string format
    date_str = selected_date.strftime("%Y-%m-%d")
    if st.button("Predict", type="primary"):
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
                    historical_val = get_historical_value(history_df, date_str)
                    
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

                    # Prediction result
                    st.subheader("Predicted Bike Rides")
                    st.metric(
                            label="Predicted Rides",
                            value=f"{int(prediction):,}",
                            delta=f"{int(prediction - historical_val):,} than in 2023"
                            )

with tab3:
    st.subheader("About the Project")
    st.markdown("""
    
    Official Citi Bike trip records are stored as multiple, large CSV files per month, with millions of rows each. 
    Therefore it can't simply be loaded into memory all at once. 
    In order to train the model and analyze the data, a big data approach was necessary.
    
    First, the raw CSV files were converted to Parquet format, which is much smaller and faster to read. 
    Then, instead of training on millions of individual trips, the data was aggregated by day, resulting in 365 daily summaries.
    This was done using Polars, which is much faster than pandas for large datasets.
                
    This approach preserves important patterns while reducing the dataset by 99.9%, making the training extremely efficient.
    
    The model is a Random Forest Regressor trained on these daily summaries, using weather data as additional features.
                
    Later, those daily summaries were also used for the historical data analysis, which revealed trends and patterns in bike usage throughout 2023.
    """)

# Footer
st.caption("Data source: NYC Citi Bike | Weather: Open-Meteo API")