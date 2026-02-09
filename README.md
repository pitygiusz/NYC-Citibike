# NYC-CitiBike-Predictor

This is a basic demand forecaster and historical data analysis for the NYC Citi Bike network, developed as a streamlit web application.

**Live Demo:** [Citi Bike Demand Predictor](https://nyc-citibike-predictor.streamlit.app/)

## Goal

The primary challenge of this project was to engineer a pipeline for large-scale data processing on consumer hardware. Dealing with 7GB+ of raw CSV data required moving beyond standard tools to overcome memory constrains and optimize storage.

## Key Features

- Big Data Engine: Processed 30M+ rows (7GB raw CSV) by converting to Parquet format and utilizing Polars for memory-efficient manipulation.

- EDA: Using vectorized operations to perform complex calculations across millions of rows in seconds.

- Machine Learning: Trained a Random Forest Regressor to estimate trip volume.

- Feature Engineering: Cyclical encoding (Sine/Cosine) for temporal features (Month, Day of Week).

- Haversine distance calculations for ride intensity analysis.

- Production: Lightweight Streamlit application integrated with the Open-Meteo API for live weather data.

- Historical Data Analysis: Interactive visualizations of 2023 bike usage patterns, including weather impact analysis and temporal trends.


## Project Structure
```
NYC-ML-Project/
├── 01_prepare_data.py        # ETL pipeline
├── 02_train_model.py         # Data aggregation and model training
├── 03_test_model.py          # Model validation
├── app.py                    # Streamlit web application
├── tools.py                  # Helper functions for the app
├── requirements.txt          # Python dependencies
└── resources/
    ├── daily_aggregated.csv  # Aggregated daily data
    └── random_forest.pkl     # Trained model
```

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/pitygiusz/NYC-Citibike
cd NYC-Citibike
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Run the full pipeline (first download data from [citibike.com](https://citibikenyc.com/system-data))
```bash
# Step 1: Prepare the data
python 01_prepare_data.py

# Step 2: Train the model
python 02_train_model.py

# Step 3: Test the model
python 03_test_model.py
```

4. Run the application:
```bash
streamlit run app.py
```



## Dataset
The dataset used for training model is the public NYC Citi Bike data, available at [citibike](https://citibikenyc.com/system-data) official website. My analysis refers to year 2023. The historical weather data is available at [NOAA](https://www.ncdc.noaa.gov/cdo-web/search).


## Technologies Used
- Python (Pandas, NumPy, Polars)
- Streamlit
- Scikit-Learn


## Contributions
This project was completed individually.
