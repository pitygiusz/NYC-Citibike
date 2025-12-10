"""
PyTorch Neural Network Models for NYC Parking Fines Prediction

This module contains neural network architectures for:
1. Regression: Predicting parking fine amounts
2. Classification: Predicting fine categories (small, medium, large)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ParkingFinesDataset(Dataset):
    """Custom Dataset for Parking Fines data"""
    
    def __init__(self, X, y):
        """
        Args:
            X: Features as numpy array or tensor
            y: Labels as numpy array or tensor
        """
        if isinstance(X, np.ndarray):
            self.X = torch.FloatTensor(X)
        else:
            self.X = X.float()
            
        if isinstance(y, np.ndarray):
            self.y = torch.FloatTensor(y)
        else:
            self.y = y.float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FineAmountRegressor(nn.Module):
    """
    Neural Network for predicting parking fine amounts (regression task)
    
    Architecture:
    - Input layer
    - 3 hidden layers with BatchNorm and Dropout
    - Output layer (1 neuron for regression)
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(FineAmountRegressor, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x).squeeze()


class FineCategoryClassifier(nn.Module):
    """
    Neural Network for predicting fine categories (classification task)
    
    Architecture:
    - Input layer
    - 3 hidden layers with BatchNorm and Dropout
    - Output layer (3 neurons for 3 categories: small, medium, large)
    """
    
    def __init__(self, input_dim, num_classes=3, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        """
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(FineCategoryClassifier, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation, will use CrossEntropyLoss)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class ModelTrainer:
    """Utility class for training PyTorch models"""
    
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: PyTorch model to train
            device: Device to use ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, optimizer, criterion, 
              num_epochs=50, early_stopping_patience=10, verbose=True):
        """
        Train the model with early stopping
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer
            criterion: Loss function
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
        
        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            X = X.to(self.device)
            predictions = self.model(X)
        return predictions.cpu().numpy()


def evaluate_regressor(model, X_test, y_test, device='cpu'):
    """
    Evaluate regression model
    
    Args:
        model: Trained PyTorch model
        X_test: Test features
        y_test: Test labels
        device: Device to use
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    trainer = ModelTrainer(model, device)
    predictions = trainer.predict(X_test)
    
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def evaluate_classifier(model, X_test, y_test, device='cpu'):
    """
    Evaluate classification model
    
    Args:
        model: Trained PyTorch model
        X_test: Test features
        y_test: Test labels (as class indices)
        device: Device to use
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    model.eval()
    with torch.no_grad():
        if isinstance(X_test, np.ndarray):
            X_test = torch.FloatTensor(X_test)
        X_test = X_test.to(device)
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)
    
    predictions = predictions.cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    report = classification_report(y_test, predictions)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'classification_report': report
    }
