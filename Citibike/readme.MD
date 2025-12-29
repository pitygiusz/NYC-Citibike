# NYC Citibike

This part of the **NYC-ML-Project** is a web application that predicts the daily number of bike rides in New York City based on weather forecasts.

**Live Demo:** [Citibike AI](https://citibike-usage-ai.onrender.com/)

## 🏆 Goal
The goal of this project was to build a reliable demand forecasting model for the NYC bike-sharing system. 

The application fetches live weather forecasts for NYC (using the Open-Meteo API), processes the data and returns a prediction along with a historical benchmark for context.

## 🚀 Key Features

* **Big Data Processing:** Efficiently handled 30M+ rows (7GB of raw CSV data) by converting them to **Parquet** format using **Polars**.
* **Machine Learning:** Trained a Random Forest regressor and saving it for future usage.
* **Feature Engineering:**
    * Cyclical encoding for months and days of week (Sine/Cosine).
    * Haversine distance calculation for every ride.
    * Integration with NOAA weather data.
* **Web App:** Lightweight Flask application with HTML frontend.

## 🛠️ Technologies used
* **Data & ML:** Python, Polars, Pandas, NumPy, Scikit-Learn 
* **Web:** Flask, Gunicorn, HTML5, JavaScript 
* **API:** pen-Meteo API 
* **Infrastructure:** Render.com 

## 📁 Dataset
The dataset used for training model is the public NYC Citi Bike data, available at [citibike](https://citibikenyc.com/system-data) official website. My analysis refers to year 2023. The historical weather data is available at [NOAA](https://www.ncdc.noaa.gov/cdo-web/search).