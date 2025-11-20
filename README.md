# NYC Parking Violations Project

## Overview
This project analyzes NYC parking violations data (Fiscal Year 2015) to understand violation patterns and predict fine categories based on vehicle and violation characteristics. The original Jupyter Notebook has been refactored into a modular Python project for better maintainability and scalability.

## Project Structure
```
.
├── run.py                  # Main entry point
├── requirements.txt        # Project dependencies
├── src/
│   ├── data_loader.py      # Data loading logic
│   ├── preprocessing.py    # Cleaning and feature engineering
│   ├── eda.py              # Exploratory Data Analysis
│   └── modeling.py         # XGBoost model training and evaluation
├── pipelines/
│   └── training_pipeline.py # Orchestration of the workflow
└── plots/                  # Generated EDA visualizations
```

## Data Source

The project requires the **NYC Parking Violations Issued - Fiscal Year 2015** dataset. You can download it from Kaggle:

[NYC Parking Tickets on Kaggle](https://www.kaggle.com/datasets/new-york-city/nyc-parking-tickets)

Please download the `Parking_Violations_Issued_-_Fiscal_Year_2015.csv` file and place it in the `archive` directory.

## Installation

1.  **Prerequisites**: Ensure you have Python 3.8+ installed.
2.  **Install Dependencies**:
    ```bash
    python3 -m pip install -r requirements.txt 
    ```

## Usage

Run the entire pipeline (Data Loading -> Preprocessing -> EDA -> Modeling):
```bash
python3 run.py
```

### Options
- **Skip EDA** (faster execution):
  ```bash
  python3 run.py --no-eda
  ```
- **Enable Hyperparameter Tuning** (GridSearchCV):
  ```bash
  python3 run.py --tune
  ```

## Analysis report
The analysis report can be found in the `EDA_REPORT.md` file. 

## Technologies used
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

The project was compleated individually.
