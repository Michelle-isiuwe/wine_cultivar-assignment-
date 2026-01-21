"""
Wine Cultivar Origin Prediction System - Model Training with Random Forest
Trains a Random Forest Classifier on the Wine dataset with 6 selected features.
Implements comprehensive multiclass classification metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


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
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    print("✓ Feature scaling completed!")
    
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
    print(f"  - Train class distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"  - Test class distribution: {dict(y_test.value_counts().sort_index())}")
    
    return X_train, X_test, y_train, y_test, scaler, selected_features


def train_model(model, X_train, y_train):
    """Train the Random Forest Classifier"""
    print("\n" + "="*70)
    print("🤖 MODEL TRAINING")
    print("="*70)
    
    print(f"\n🎓 Training Configuration:")
    print(f"  - Algorithm: Random Forest Classifier")
    print(f"  - Number of Trees: 100")
    print(f"  - Max Depth: 15")
    print(f"  - Random State: 42")
    
    print(f"\n🎯 Training Progress...")
    model.fit(X_train, y_train)
    print(f"✓ Training completed!")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Feature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']:<25} : {row['importance']:.4f}")


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model with comprehensive metrics"""
    print("\n" + "="*70)
    print("📈 MODEL EVALUATION - COMPREHENSIVE METRICS")
    print("="*70)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # ACCURACY
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n🎯 ACCURACY METRICS:")
    print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # PRECISION, RECALL, F1-SCORE
    precision_macro = precision_score(y_test, y_test_pred, average='macro')
    precision_weighted = precision_score(y_test, y_test_pred, average='weighted')
    recall_macro = recall_score(y_test, y_test_pred, average='macro')
    recall_weighted = recall_score(y_test, y_test_pred, average='weighted')
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    
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
    precision_per_class = precision_score(y_test, y_test_pred, average=None)
    recall_per_class = recall_score(y_test, y_test_pred, average=None)
    f1_per_class = f1_score(y_test, y_test_pred, average=None)
    support_per_class = np.bincount(y_test)
    
    print(f"\n   Per-Class Metrics:")
    print(f"   {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print(f"   {'-'*60}")
    for i in range(3):
        print(f"   Cultivar {i:<2} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f} {support_per_class[i]:<8}")
    
    # Classification Report
    print(f"\n📋 CLASSIFICATION REPORT:")
    print("-" * 70)
    report = classification_report(
        y_test, y_test_pred,
        target_names=['Cultivar 0', 'Cultivar 1', 'Cultivar 2'],
        digits=4
    )
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
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


def save_model(model, scaler, model_path='wine_model.pkl', scaler_path='scaler.pkl'):
    """Save the trained model and scaler"""
    print(f"\n💾 Saving Model and Scaler...")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {scaler_path}")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🍷 WINE CULTIVAR ORIGIN PREDICTION SYSTEM")
    print("🌳 Random Forest Classifier Implementation")
    print("="*70)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, selected_features = load_data()
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"\n📐 Model Configuration:")
    print(f"  - Algorithm: Random Forest Classifier")
    print(f"  - Number of Trees: 100")
    print(f"  - Max Depth: 15")
    print(f"  - Number of Jobs: -1 (all processors)")
    
    # Train model
    train_model(model, X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Save model and scaler
    save_model(model, scaler)
    
    print(f"\n✅ Model training and evaluation completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
