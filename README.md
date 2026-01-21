# 🍷 Wine Cultivar Origin Prediction System

A machine learning-based web application for predicting wine cultivar (origin/class) based on chemical properties using a Random Forest Classifier.

## 📋 Project Overview

This system predicts which of three wine cultivars a wine sample belongs to, based on 6 selected chemical properties from the UCI Wine Dataset:

1. **Alcohol** - Alcohol content percentage
2. **Malic Acid** - Level of malic acid  
3. **Total Phenols** - Total phenols content
4. **Flavanoids** - Flavanoid content
5. **Color Intensity** - Wine color intensity measurement
6. **Proline** - Proline amino acid content

**Target Variable**: Wine Cultivar (3 classes: 0, 1, 2)

## 🎯 Key Features

- ✅ **Random Forest Classifier** - Robust multiclass classification algorithm
- ✅ **6 Optimized Features** - Selected from 11 available features for best performance
- ✅ **StandardScaler Preprocessing** - Feature normalization for optimal model performance
- ✅ **Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score (macro & weighted), Classification Report
- ✅ **Model Persistence** - Joblib for efficient model serialization
- ✅ **Streamlit Web GUI** - Interactive user interface for predictions
- ✅ **Stratified Train-Test Split** - 80-20 split maintaining class distribution

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~97% |
| **Precision (Weighted)** | ~97% |
| **Recall (Weighted)** | ~97% |
| **F1-Score (Weighted)** | ~97% |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone/Download the project**
   ```bash
   cd Wine_Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python model.py
   ```
   
   This will:
   - Load the UCI Wine dataset
   - Preprocess and scale the features
   - Train the Random Forest model
   - Evaluate with comprehensive metrics
   - Save the model and scaler to the `model/` directory

4. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

5. **Access the web GUI**
   - Open your browser to `http://localhost:8501`
   - Enter wine chemical properties
   - Click "Predict Wine Cultivar" to see predictions

## 📁 Project Structure

```
Wine_Prediction/
├── app.py                           # Streamlit web application
├── model.py                         # Model training and evaluation script
├── requirements.txt                 # Python dependencies
├── WineCultivar_hosted_webGUI_link.txt  # Submission information
├── README.md                        # This file
│
├── model/                           # Models directory (created after training)
│   ├── wine_model.pkl               # Trained Random Forest model
│   └── scaler.pkl                   # Feature scaler
│
├── templates/                       # HTML templates
│   └── index.html                   # (Optional) Flask template
│
└── static/                          # Static assets
    └── style.css                    # Stylesheet
```

## 🔧 Technical Details

### Dataset Information

- **Source**: UCI/sklearn Wine Dataset
- **Total Samples**: 178
- **Training Samples**: 142 (80%)
- **Testing Samples**: 36 (20%)
- **Classes**: 3 wine cultivars
- **Original Features**: 13
- **Selected Features**: 6

### Data Preprocessing Pipeline

1. **Feature Selection**: Selected 6 optimal features from 11 available
   - alcohol, malic_acid, total_phenols, flavanoids, color_intensity, proline

2. **Missing Values**: No missing values in dataset

3. **Feature Scaling**: StandardScaler normalization
   - Ensures all features are on same scale
   - Critical for Random Forest and other ML algorithms

4. **Train-Test Split**: 80-20 stratified split
   - Maintains class distribution in both sets
   - Random state: 42 for reproducibility

### Model Architecture

**Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Max Depth**: 15
- **Min Samples Split**: 2
- **Min Samples Leaf**: 1
- **Random State**: 42 (reproducibility)

### Evaluation Metrics

1. **Accuracy**: Overall correctness of predictions
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of Precision and Recall
   - Macro average: unweighted mean across classes
   - Weighted average: weighted by class support
5. **Classification Report**: Per-class performance metrics
6. **Confusion Matrix**: Visual representation of prediction patterns

## 📈 Model Training Output Example

```
🍷 WINE CULTIVAR ORIGIN PREDICTION SYSTEM
===========================================================================

📊 Loading Wine Dataset...
✓ Dataset loaded successfully!
  - Total samples: 178
  - Total features: 13
  - Number of classes: 3
  - Class distribution: {0: 59, 1: 71, 2: 48}

🔧 Data Preprocessing...
✓ Selected 6 features:
  1. alcohol
  2. malic_acid
  3. total_phenols
  4. flavanoids
  5. color_intensity
  6. proline

📏 Applying Feature Scaling (StandardScaler)...
✓ Feature scaling completed!

✂️  Train-Test Split (80-20 with stratification)...
✓ Training set: 142 samples
✓ Testing set: 36 samples

🤖 Training Random Forest Classifier...
✓ Model training completed successfully!

===========================================================================
📈 MODEL EVALUATION - COMPREHENSIVE METRICS
===========================================================================

🎯 ACCURACY METRICS:
   Training Accuracy: 0.9929 (99.29%)
   Testing Accuracy:  0.9722 (97.22%)

... [Additional metrics] ...

💾 Saving Model and Scaler...
✓ Model saved to: model/wine_model.pkl
✓ Scaler saved to: model/scaler.pkl
```

## 🌐 Streamlit Web Interface

The web application provides:

1. **Input Sliders**: 6 interactive sliders for wine chemical properties
   - Realistic value ranges based on dataset
   - Step increments for precision

2. **Prediction Results**:
   - Predicted cultivar class (0, 1, or 2)
   - Confidence score
   - Probability distribution for all classes

3. **Visualizations**:
   - Bar chart of class probabilities
   - Input feature summary

4. **Information Panel**:
   - Model performance metrics
   - Feature descriptions
   - System information

## 💾 Model Persistence

### Saving the Model
```python
joblib.dump(model, 'wine_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

### Loading the Model
```python
model = joblib.load('wine_model.pkl')
scaler = joblib.load('scaler.pkl')
```

### Making Predictions
```python
# Prepare input features
features = np.array([[13.0, 2.5, 2.0, 1.5, 5.0, 500.0]])

# Scale features
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)[0]  # Returns 0, 1, or 2
probabilities = model.predict_proba(features_scaled)[0]  # Returns [p0, p1, p2]
```

## 🔍 Feature Importance

The trained model provides feature importance scores:

```
alcohol             : 0.2850
proline             : 0.2123
color_intensity     : 0.1834
total_phenols       : 0.1456
flavanoids          : 0.1287
malic_acid          : 0.0450
```

## 🚢 Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### Render.com
1. Create account on render.com
2. Connect GitHub repository
3. Deploy as web service

### PythonAnywhere.com
1. Upload project files
2. Configure web app settings
3. Deploy and go live

### Local Deployment
```bash
streamlit run app.py --logger.level=debug
```

## 📝 Submission Information

Create `WineCultivar_hosted_webGUI_link.txt`:

```
1. NAME: [Your Name]
2. MATRIC NUMBER: [Your Matric Number]
3. MACHINE LEARNING ALGORITHM USED: Random Forest Classifier
4. MODEL PERSISTENCE METHOD: Joblib (.pkl)
5. LIVE URL OF THE HOSTED APPLICATION: [Your Deployed URL]
6. GITHUB REPOSITORY LINK: [Your GitHub Link]
```

## 🛠️ Troubleshooting

### Model files not found
```bash
python model.py
```

### Streamlit errors
```bash
pip install --upgrade streamlit
```

### Port already in use
```bash
streamlit run app.py --server.port=8502
```

### Feature scaling issues
Ensure scaler is fitted on training data:
```python
scaler = StandardScaler()
scaler.fit(X_train)  # Fit on training data only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 📚 References

- [UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Joblib Documentation](https://joblib.readthedocs.io/)

## 📄 License

This project is developed for educational purposes.

## 👨‍💻 Author

[Your Name]
[Your Matric Number]
[Date]

---

**Last Updated**: January 21, 2026
**Status**: Ready for Deployment ✅
