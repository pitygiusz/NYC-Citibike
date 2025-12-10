"""
Training Script for PyTorch Neural Network Models
Trains both regression and classification models for NYC Parking Fines prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

from pytorch_model import (
    FineAmountRegressor, 
    FineCategoryClassifier,
    ParkingFinesDataset,
    ModelTrainer,
    evaluate_regressor,
    evaluate_classifier
)


def load_and_preprocess_data(data_path='data/Parking_Violations_Issued_-_Fiscal_Year_2015.csv',
                              fine_codes_path='data/ParkingViolationCodes.csv',
                              sample_size=100000):
    """
    Load and preprocess parking fines data
    
    Args:
        data_path: Path to main dataset
        fine_codes_path: Path to violation codes dataset
        sample_size: Number of samples to use for training
    
    Returns:
        Tuple of (X, y_regression, y_classification)
    """
    print("Loading data...")
    
    # Load data
    fine_amount = pd.read_csv(fine_codes_path)
    df = pd.read_csv(data_path, engine='pyarrow')
    
    # Join with fine amounts
    df = df.join(fine_amount, on='Violation Code', how='inner')
    
    # Drop unnecessary columns
    columns_to_drop = [
        'Street Code1', 'Street Code2', 'Street Code3', 'Violation Location', 'Violation Precinct',
        'Issuer Precinct', 'Issuer Code', 'Issuer Command', 'Issuer Squad', 'Time First Observed',
        'Law Section', 'Issuing Agency',
        'Sub Division', 'Violation Legal Code', 'Days Parking In Effect    ',
        'Unregistered Vehicle?', 'Meter Number',
        'Feet From Curb', 'Violation Post Code', 'Violation Description',
        'No Standing or Stopping Violation', 'Hydrant Violation',
        'Double Parking Violation', 'Latitude', 'Longitude', 'Community Board',
        'Community Council ', 'Census Tract', 'BIN', 'BBL', 'NTA', 'VIOLATION CODE', 'VIOLATION DESCRIPTION'
    ]
    df = df.drop(columns=columns_to_drop)
    
    print("Preprocessing data...")
    
    # Process Violation Time
    vt = df['Violation Time'].str.upper().str.replace('A', 'AM').str.replace('P', 'PM')
    vt = vt.str[:2] + ':' + vt.str[2:4] + vt.str[4:]
    vt = vt.str.replace('^00', '12', regex=True)
    df['Violation Hour'] = pd.to_datetime(vt, format='%I:%M%p', errors='coerce')
    df['Violation_Hour_Num'] = df['Violation Hour'].dt.hour
    
    # Encode hour as cyclical features
    df['Hour_sin'] = np.sin(2 * np.pi * df['Violation_Hour_Num'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Violation_Hour_Num'] / 24)
    
    # Process dates
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], format="%m/%d/%Y")
    df['Vehicle Expiration'] = df['Vehicle Expiration Date'].str[:10]
    df['Vehicle Expiration'] = pd.to_datetime(df['Vehicle Expiration'], format="%m/%d/%Y", errors='coerce')
    
    # Create fine categories for classification
    bins = [0, 50, 100, np.inf]
    labels = ['small', 'medium', 'large']
    df['Fine Category'] = pd.cut(df['Fine Amount $'], bins=bins, labels=labels, right=True)
    
    # Define features
    categorical_cols = [
        "Registration State", "Plate Type", "Vehicle Body Type", "Vehicle Make",
        "Violation County", "Violation In Front Of Or Opposite", "Plate ID"
    ]
    numeric_cols = [
        "Vehicle Expiration", "Vehicle Year", "Hour_sin", "Hour_cos", "Issue Date"
    ]
    
    # Drop rows with missing values
    df = df.dropna(subset=categorical_cols + numeric_cols)
    df = df[df['Fine Amount $'] != 0]
    
    # Sample data
    print(f"Sampling {sample_size} records...")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Prepare features
    X = df_sample[categorical_cols + numeric_cols].copy()
    
    # Extract date components
    for col in ["Vehicle Expiration", "Issue Date"]:
        X[col + "_year"] = X[col].dt.year
        X[col + "_month"] = X[col].dt.month
        X[col + "_day"] = X[col].dt.day
    
    X = X.drop(columns=["Vehicle Expiration", "Issue Date"])
    
    # Prepare targets
    y_regression = df_sample['Fine Amount $'].values
    y_classification = df_sample['Fine Category'].values
    
    return X, y_regression, y_classification


def preprocess_features(X_train, X_test):
    """
    Preprocess features using StandardScaler and OneHotEncoder
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        Tuple of (X_train_processed, X_test_processed, preprocessor)
    """
    print("Preprocessing features...")
    
    # Identify categorical and numeric columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    num_cols = X_train.select_dtypes(exclude=["object"]).columns
    
    # Create preprocessor
    # Note: with_mean=False is used to match the original notebook's preprocessing pipeline
    # This ensures compatibility and fair comparison with existing models
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="passthrough"
    )
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, preprocessor


def train_regression_model(X_train, X_test, y_train, y_test, 
                           device='cpu', save_path='models/pytorch_regressor.pth'):
    """
    Train regression model for fine amount prediction
    
    Args:
        X_train, X_test: Train and test features
        y_train, y_test: Train and test targets
        device: Device to use
        save_path: Path to save model
    
    Returns:
        Trained model and evaluation metrics
    """
    print("\n" + "="*60)
    print("Training Regression Model (Fine Amount Prediction)")
    print("="*60)
    
    input_dim = X_train.shape[1]
    
    # Create model
    model = FineAmountRegressor(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3
    )
    
    # Create datasets and dataloaders
    train_dataset = ParkingFinesDataset(X_train, y_train)
    test_dataset = ParkingFinesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train
    trainer = ModelTrainer(model, device)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=50,
        early_stopping_patience=10,
        verbose=True
    )
    
    # Evaluate
    metrics = evaluate_regressor(model, X_test, y_test, device)
    
    print("\nRegression Model Evaluation Metrics:")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'metrics': metrics,
        'history': history
    }, save_path)
    print(f"\nModel saved to {save_path}")
    
    return model, metrics


def train_classification_model(X_train, X_test, y_train, y_test,
                               device='cpu', save_path='models/pytorch_classifier.pth'):
    """
    Train classification model for fine category prediction
    
    Args:
        X_train, X_test: Train and test features
        y_train, y_test: Train and test targets
        device: Device to use
        save_path: Path to save model
    
    Returns:
        Trained model and evaluation metrics
    """
    print("\n" + "="*60)
    print("Training Classification Model (Fine Category Prediction)")
    print("="*60)
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)
    
    # Create model
    model = FineCategoryClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3
    )
    
    # Create datasets and dataloaders
    train_dataset = ParkingFinesDataset(X_train, y_train_enc)
    test_dataset = ParkingFinesDataset(X_test, y_test_enc)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train
    trainer = ModelTrainer(model, device)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=50,
        early_stopping_patience=10,
        verbose=True
    )
    
    # Evaluate
    metrics = evaluate_classifier(model, X_test, y_test_enc, device)
    
    print("\nClassification Model Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'label_encoder': le,
        'metrics': metrics,
        'history': history
    }, save_path)
    print(f"\nModel saved to {save_path}")
    
    return model, metrics


def main():
    """Main training pipeline"""
    
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X, y_regression, y_classification = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.3, random_state=42
    )
    
    # Use same split for classification
    _, _, y_class_train, y_class_test = train_test_split(
        X, y_classification, test_size=0.3, random_state=42
    )
    
    # Preprocess features
    X_train_processed, X_test_processed, preprocessor = preprocess_features(X_train, X_test)
    
    # Save preprocessor
    os.makedirs('models', exist_ok=True)
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Preprocessor saved to models/preprocessor.pkl")
    
    # Train regression model
    reg_model, reg_metrics = train_regression_model(
        X_train_processed, X_test_processed,
        y_reg_train, y_reg_test,
        device=device
    )
    
    # Train classification model
    class_model, class_metrics = train_classification_model(
        X_train_processed, X_test_processed,
        y_class_train, y_class_test,
        device=device
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nSummary:")
    print(f"Regression Model - MAE: {reg_metrics['mae']:.2f}, RMSE: {reg_metrics['rmse']:.2f}, R²: {reg_metrics['r2']:.4f}")
    print(f"Classification Model - Accuracy: {class_metrics['accuracy']:.4f}, F1: {class_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
