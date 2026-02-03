import polars as pl
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time
from sklearn.model_selection import train_test_split


DATA_PATH_PATTERN = "data/2023-citibike-tripdata/*.parquet" #Path to Parquet files
WEATHER_PATH = "data/4180612.csv" #Path to weather data

def haversine_expr(lat1_col, lon1_col, lat2_col, lon2_col):
    R = 6371
    lat1 = lat1_col * (np.pi / 180)
    lon1 = lon1_col * (np.pi / 180)
    lat2 = lat2_col * (np.pi / 180)
    lon2 = lon2_col * (np.pi / 180)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = (dlat / 2).sin().pow(2) + lat1.cos() * lat2.cos() * (dlon / 2).sin().pow(2)
    c = 2 * a.sqrt().arcsin()
    return c * R

def main():
    print("--- Starting Pipeline ---")
    start_total = time.time()

    print("Loading weather data...")
    weather = (
        pl.read_csv(WEATHER_PATH)
        .rename({'DATE': 'date', 'AWND': 'wind_avg', 'PRCP': 'percip_total', 'TMIN': 'temp_min', 'TMAX': 'temp_max'})
        .select(['date', 'wind_avg', 'percip_total', 'temp_min', 'temp_max'])
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    )

    #farenheit to celsius
    weather = weather.with_columns([
        ((pl.col("temp_min") - 32) * 5.0/9.0).alias("temp_min"),
        ((pl.col("temp_max") - 32) * 5.0/9.0).alias("temp_max"),
        (pl.col("wind_avg") * 1.60934).alias("wind_avg")
    ])


    print("Scanning Parquet files...")
    q = pl.scan_parquet(DATA_PATH_PATTERN)

    print("Transforming and aggregating data...")
    q = (
        q

        .with_columns(
            pl.col("started_at").str.to_datetime("%Y-%m-%d %H:%M:%S.%f").alias("dt_obj")
        )
        .with_columns([
            pl.col("dt_obj").dt.date().alias("date"),
            pl.col("dt_obj").dt.month().alias("month"),
            pl.col("dt_obj").dt.weekday().alias("day_of_week")
        ])

        .with_columns([
            (pl.col("day_of_week") * 2 * np.pi / 7).alias("day_angle"),
            ((pl.col("month") - 1) * 2 * np.pi / 12).alias("month_angle")
        ])
        .with_columns([
            pl.col("day_angle").sin().alias("day_of_week_sin"),
            pl.col("day_angle").cos().alias("day_of_week_cos"),
            pl.col("month_angle").sin().alias("month_sin"),
            pl.col("month_angle").cos().alias("month_cos"),

            haversine_expr(
                pl.col("start_lat"), pl.col("start_lng"), 
                pl.col("end_lat"), pl.col("end_lng")
            ).alias("distance_km")
        ])

        .filter(pl.col("date") >= pl.date(2023, 1, 1))

        .group_by("date")
        .agg([
            pl.len().alias("ride_id_count"),
            pl.col("distance_km").mean().alias("distance_km_mean"),
            pl.col("month_sin").first(),
            pl.col("month_cos").first(),
            pl.col("day_of_week_sin").first(),
            pl.col("day_of_week_cos").first()
        ])
    )

    daily_agg = q.collect(streaming=True)

    print(f"Aggregation done. Result shape: {daily_agg.shape}")
    
    daily = (
        daily_agg
        .join(weather, on='date', how='left')
        .drop_nulls()
        .sort("date")
    )
    

    daily.write_csv('resources/daily_aggregated.csv') # Save aggregated data

    print("Training Random Forest...")
    
    df_ml = daily.to_pandas() # Convert small aggregated data to pandas for sklearn
    features = ['month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 'wind_avg', 'percip_total', 'temp_max']
    target = 'ride_id_count'

    print("Training Random Forest model...")

    features = ['month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 'wind_avg', 'percip_total', 'temp_max']
    target = 'ride_id_count'
    X = daily[features]
    y = daily[target]


    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)


    print("Random Forest Model Training Pipeline Completed.")

    joblib.dump(model, 'resources/random_forest.pkl') # Save the trained model
    print(f"Pipeline completed in {time.time() - start_total:.2f} seconds.")

if __name__ == "__main__":
    main()