# NYC-ML-Project

## Project Overview
A data science project focused on NYC urban infrastructure, processing 30M+ records of real world data. It features a production-ready demand forecaster for Citibike and a deep dive into statistical analysis of NYC Parking Violations, emphasizing high-performance data processing and model accuracy.

## NYC Citibike: Demand Forecating

**Live Demo:** [Citibike AI](https://citibike-usage-ai.onrender.com/)

A web application that predicts the daily number of bike rides in NYC based on real-time weather forecasts.

### Key Features

- Big Data Engine: Processed 30M+ rows (7GB raw CSV) by converting to Parquet format and utilizing Polars for memory-efficient manipulation.

- Machine Learning: Trained a Random Forest Regressor to estimate trip volume.

- Feature Engineering: * Cyclical encoding (Sine/Cosine) for temporal features (Month, Day of Week).

- Haversine distance calculations for ride intensity analysis.

- Production: Lightweight Flask API integrated with the Open-Meteo API for live weather data.

### Dataset
The dataset used for training model is the public NYC Citi Bike data, available at [citibike](https://citibikenyc.com/system-data) official website. My analysis refers to year 2023. The historical weather data is available at [NOAA](https://www.ncdc.noaa.gov/cdo-web/search).


## NYC Parking Fines: Enforcement Analytics

An analytical framework focused on identifying patterns in parking violations and predicting fine categories based on vehicle and geographical data.

### Key Features

- Data Preprocessing: Cleaning and filtering of Fiscal Year 2015 violation data.

- Exploratory Data Analysis (EDA): Visualizing violation density by borough, time of day, and vehicle make.

- Hypothesis Testing: Statistical validation of enforcement trends across different demographics.

- Predictive Modeling: Utilizing XGBoost to classify fine amounts based on violation characteristics.

### Dataset
The dataset used is provided publicly by City of New York and is available at [NYC OpenData](https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2023/869v-vr48/about_data) website. My analysis refers to fiscal year 2015.


## Technologies Used
- Python (Pandas, NumPy, Polars, Scikit-Learn , XGBoost)
- Matplotlib, Seaborn
- Flask, Gunicorn, HTML5, JavaScript 
- Render.com
- Jupyter Notebook


## Contributions
This project was completed individually.
