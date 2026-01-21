"""
Wine Cultivar Origin Prediction System - Model Training with PyTorch
Trains a Neural Network on the Wine dataset with 6 selected features.
Implements comprehensive multiclass classification metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class WineNeuralNetwork(nn.Module):
    """Neural Network for Wine Cultivar Classification"""
    
    def __init__(self, input_size=6, hidden_size1=64, hidden_size2=32, num_classes=3):
        super(WineNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size2, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


def load_data():
    """Load and preprocess the Wine dataset"""
    print("\n" + "="*70)
    print("📊 LOADING AND PREPROCESSING DATA")
    print("="*70)
    
    # Load dataset
    print("\n📊 Loading Wine Dataset...")
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name='cultivar')
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Total samples: {X.shape[0]}")
    print(f"  - Total features: {X.shape[1]}")
    print(f"  - Number of classes: {len(np.unique(y))}")
    
    # Select 6 features
    selected_features = [
        'alcohol',
        'malic_acid',
        'total_phenols',
        'flavanoids',
        'color_intensity',
        'proline'
    ]
    
    X_selected = X[selected_features].copy()
    
    print(f"\n✓ Selected 6 Features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
    
    # Feature scaling
    print(f"\n📏 Applying Feature Scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Train-test split with stratification
    print(f"\n✂️  Train-Test Split (80-20 with stratification)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Testing set: {X_test.shape[0]} samples")
    print(f"  - Train class distribution: {dict(pd.Series(y_train).value_counts().sort_index())}")
    print(f"  - Test class distribution: {dict(pd.Series(y_test).value_counts().sort_index())}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train.values).to(device)
    y_test_tensor = torch.LongTensor(y_test.values).to(device)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler, selected_features


def train_model(model, train_loader, test_loader, epochs=200, learning_rate=0.001):
    """Train the neural network"""
    print("\n" + "="*70)
    print("🤖 MODEL TRAINING")
    print("="*70)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n🎓 Training Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Optimizer: Adam")
    print(f"  - Loss Function: CrossEntropyLoss")
    
    train_losses = []
    test_losses = []
    
    print(f"\n🎯 Training Progress:")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    print(f"\n✓ Training completed!")
    
    return train_losses, test_losses


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model with comprehensive metrics"""
    print("\n" + "="*70)
    print("📈 MODEL EVALUATION - COMPREHENSIVE METRICS")
    print("="*70)
    
    model.eval()
    
    with torch.no_grad():
        # Training predictions
        train_outputs = model(X_train)
        train_pred = torch.argmax(train_outputs, dim=1).cpu().numpy()
        
        # Testing predictions
        test_outputs = model(X_test)
        test_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
    
    y_train_np = y_train.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    # ACCURACY
    train_accuracy = accuracy_score(y_train_np, train_pred)
    test_accuracy = accuracy_score(y_test_np, test_pred)
    
    print(f"\n🎯 ACCURACY METRICS:")
    print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # PRECISION, RECALL, F1-SCORE
    precision_macro = precision_score(y_test_np, test_pred, average='macro')
    precision_weighted = precision_score(y_test_np, test_pred, average='weighted')
    recall_macro = recall_score(y_test_np, test_pred, average='macro')
    recall_weighted = recall_score(y_test_np, test_pred, average='weighted')
    f1_macro = f1_score(y_test_np, test_pred, average='macro')
    f1_weighted = f1_score(y_test_np, test_pred, average='weighted')
    
    print(f"\n📊 PRECISION, RECALL, F1-SCORE (Test Set):")
    print(f"\n   Macro Averages (unweighted):")
    print(f"     Precision: {precision_macro:.4f}")
    print(f"     Recall:    {recall_macro:.4f}")
    print(f"     F1-Score:  {f1_macro:.4f}")
    
    print(f"\n   Weighted Averages (by class support):")
    print(f"     Precision: {precision_weighted:.4f}")
    print(f"     Recall:    {recall_weighted:.4f}")
    print(f"     F1-Score:  {f1_weighted:.4f}")
    
    # Per-class metrics
    precision_per_class = precision_score(y_test_np, test_pred, average=None)
    recall_per_class = recall_score(y_test_np, test_pred, average=None)
    f1_per_class = f1_score(y_test_np, test_pred, average=None)
    support_per_class = np.bincount(y_test_np)
    
    print(f"\n   Per-Class Metrics:")
    print(f"   {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print(f"   {'-'*60}")
    for i in range(3):
        print(f"   Cultivar {i:<2} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f} {support_per_class[i]:<8}")
    
    # Classification Report
    print(f"\n📋 CLASSIFICATION REPORT:")
    print("-" * 70)
    report = classification_report(
        y_test_np, test_pred,
        target_names=['Cultivar 0', 'Cultivar 1', 'Cultivar 2'],
        digits=4
    )
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_np, test_pred)
    print(f"🎯 CONFUSION MATRIX:")
    print(cm)
    
    # Visualize Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cultivar 0', 'Cultivar 1', 'Cultivar 2'])
    disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Confusion Matrix (Count)', fontsize=12, fontweight='bold')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['Cultivar 0', 'Cultivar 1', 'Cultivar 2'])
    disp2.plot(ax=axes[1], cmap='Greens', values_format='.2%')
    axes[1].set_title('Confusion Matrix (Normalized %)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Confusion matrix saved to confusion_matrix.png")
    print("="*70)


def save_model(model, scaler, model_path='wine_model.pth', scaler_path='scaler.pkl'):
    """Save the trained model and scaler"""
    print(f"\n💾 Saving Model and Scaler...")
    
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {scaler_path}")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🍷 WINE CULTIVAR ORIGIN PREDICTION SYSTEM")
    print("🔧 PyTorch Neural Network Implementation")
    print("="*70)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, selected_features = load_data()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = WineNeuralNetwork(input_size=6, hidden_size1=64, hidden_size2=32, num_classes=3).to(device)
    
    print(f"\n📐 Model Architecture:")
    print(f"  - Input Layer: 6 features")
    print(f"  - Hidden Layer 1: 64 neurons + BatchNorm + ReLU + Dropout(0.3)")
    print(f"  - Hidden Layer 2: 32 neurons + BatchNorm + ReLU + Dropout(0.3)")
    print(f"  - Output Layer: 3 neurons (cultivars)")
    
    # Train model
    train_losses, test_losses = train_model(model, train_loader, test_loader, epochs=200, learning_rate=0.001)
    
    # Evaluate model
    evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Save model and scaler
    save_model(model, scaler)
    
    print(f"\n✅ Model training and evaluation completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
