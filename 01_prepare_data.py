import os
import glob
import zipfile
import polars as pl
import shutil
import time

DATA_FOLDER = "data/2023-citibike-tripdata"

def ETL_pipeline():
    
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Folder '{DATA_FOLDER}' does not exist.")
        return

    existing_csvs = glob.glob(os.path.join(DATA_FOLDER, "**/*.csv"), recursive=True) # Find existing CSVs
    for f in existing_csvs: 
        try:
            os.remove(f) # Remove existing CSV files
        except: pass
    
    zip_files = glob.glob(os.path.join(DATA_FOLDER, "*.zip")) # Find all ZIP files
    print(f"Found {len(zip_files)} ZIP files to process.")

    total_files_converted = 0
    start_time = time.time()

    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as z: # Open the ZIP file
                all_files_in_zip = z.namelist() 
                
                valid_csvs = [
                    f for f in all_files_in_zip 
                    if f.endswith(".csv") 
                ]

                if not valid_csvs:
                    print(f"Skipped {os.path.basename(zip_path)}: No valid CSV files found.")
                    continue

                print(f"Processing ZIP: {os.path.basename(zip_path)} (Found CSV: {len(valid_csvs)})")

                for csv_filename in valid_csvs:
                    clean_name = os.path.basename(csv_filename).replace(".csv", ".parquet") # Clean CSV name to create Parquet name
                    final_parquet_path = os.path.join(DATA_FOLDER, clean_name) # Final Parquet file path

                    print(f"Extracting {csv_filename}...")
                    z.extract(csv_filename, DATA_FOLDER) # Extract CSV to data folder
                    
                    extracted_csv_path = os.path.join(DATA_FOLDER, csv_filename)

                    print(f"Converting {clean_name} ...")
                    
                    try:
                        pl.scan_csv( 
                            extracted_csv_path,
                            ignore_errors=True,   
                            infer_schema_length=10000,
                            schema_overrides={
                                "start_station_id": pl.String, 
                                "end_station_id": pl.String,  
                                "ride_id": pl.String,
                                "start_station_name": pl.String,
                                "end_station_name": pl.String
                            }
                        ).sink_parquet(final_parquet_path) # Convert CSV to Parquet
                        
                        total_files_converted += 1
                        
                        os.remove(extracted_csv_path) # Remove the extracted CSV file
                        
                        csv_dir = os.path.dirname(extracted_csv_path)
                        if csv_dir and csv_dir != DATA_FOLDER: 
                            try:
                                os.rmdir(csv_dir) 
                            except: pass

                    except Exception as e:
                        print(f"Error converting {csv_filename}: {e}")
                        if os.path.exists(extracted_csv_path):
                            os.remove(extracted_csv_path)

        except Exception as e:
            print(f"Error {zip_path}: {e}")

    elapsed = time.time() - start_time
    print("-" * 40)
    print(f"Finished. Files converted: {total_files_converted}")
    print(f"Duration: {elapsed:.2f} s")
    
    final_parquets = glob.glob(os.path.join(DATA_FOLDER, "*.parquet"))
    print(f"There are {len(final_parquets)} .parquet files ready to use.")

if __name__ == "__main__":
    ETL_pipeline()