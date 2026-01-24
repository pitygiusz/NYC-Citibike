# NYC-CitiBike-Predictor

This is a production-ready demand forecaster for the NYC Citi Bike network, developed as a full-stack web application.

**Live Demo:** [Citibike AI](https://citibike-usage-ai.onrender.com/)

## Goal

The primary challenge of this project was to engineer a pipeline for large-scale data processing on consumer hardware. Dealing with 7GB+ of raw CSV data required moving beyond standard tools to overcome memory constrains and optimize storage.

### Key Features

- Big Data Engine: Processed 30M+ rows (7GB raw CSV) by converting to Parquet format and utilizing Polars for memory-efficient manipulation.

- EDA: Using vectorized operations to perform complex calculations across millions of rows in seconds.

- Machine Learning: Trained a Random Forest Regressor to estimate trip volume.

- Feature Engineering: Cyclical encoding (Sine/Cosine) for temporal features (Month, Day of Week).

- Haversine distance calculations for ride intensity analysis.

- Production: Lightweight Flask API integrated with the Open-Meteo API for live weather data.

### Dataset
The dataset used for training model is the public NYC Citi Bike data, available at [citibike](https://citibikenyc.com/system-data) official website. My analysis refers to year 2023. The historical weather data is available at [NOAA](https://www.ncdc.noaa.gov/cdo-web/search).


## Technologies Used
- Python (Pandas, NumPy, Polars, Scikit-Learn)
- Flask, Gunicorn, HTML5, JavaScript 
- Render.com
- Jupyter Notebook


## Contributions
This project was completed individually.
