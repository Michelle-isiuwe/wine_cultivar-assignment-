"""
🍷 Wine Cultivar Origin Prediction System - Streamlit Web GUI
Predicts wine cultivar (origin/class) based on 6 chemical properties
Using PyTorch Neural Network trained on 6 selected features
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Wine Cultivar Prediction",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Define the Neural Network Model
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

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        # Initialize model
        device = torch.device('cpu')
        model = WineNeuralNetwork(input_size=6, hidden_size1=64, hidden_size2=32, num_classes=3)
        
        # Load model weights
        model_path = "wine_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        else:
            st.error("❌ Model file not found! Please run train_model.py first.")
            st.stop()
        
        # Load scaler
        scaler_path = "scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            st.error("❌ Scaler file not found! Please run train_model.py first.")
            st.stop()
        
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load model and scaler
try:
    model, scaler = load_model_and_scaler()
    model_loaded = True
except:
    model_loaded = False

# Title and description
st.title("🍷 Wine Cultivar Origin Prediction System")
st.markdown("""
---
### 🎯 Predict Wine Cultivar Classification

Enter the chemical properties of your wine sample to predict its **cultivar (origin/class)**.

The model uses a **PyTorch Neural Network** trained on 6 key wine chemical features:
1. **Alcohol** - Alcohol content percentage
2. **Malic Acid** - Level of malic acid
3. **Total Phenols** - Total phenols content
4. **Flavanoids** - Flavanoid content
5. **Color Intensity** - Wine color intensity
6. **Proline** - Proline amino acid content

---
""")

if model_loaded:
    # Create columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alcohol = st.slider(
            "Alcohol Content (%)",
            min_value=11.0,
            max_value=14.9,
            value=13.0,
            step=0.1,
            help="Typical wine alcohol range"
        )
    
    with col2:
        malic_acid = st.slider(
            "Malic Acid",
            min_value=0.0,
            max_value=5.5,
            value=2.5,
            step=0.1,
            help="Acidity level"
        )
    
    with col3:
        total_phenols = st.slider(
            "Total Phenols",
            min_value=0.0,
            max_value=3.9,
            value=2.0,
            step=0.1,
            help="Phenol compounds"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        flavanoids = st.slider(
            "Flavanoids",
            min_value=0.0,
            max_value=5.1,
            value=1.5,
            step=0.1,
            help="Flavanoid content"
        )
    
    with col5:
        color_intensity = st.slider(
            "Color Intensity",
            min_value=0.0,
            max_value=13.0,
            value=5.0,
            step=0.1,
            help="Wine color intensity"
        )
    
    with col6:
        proline = st.slider(
            "Proline",
            min_value=0.0,
            max_value=1680.0,
            value=500.0,
            step=10.0,
            help="Proline amino acid content"
        )
    
    st.markdown("---")
    
    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button(
            "🔍 Predict Wine Cultivar",
            use_container_width=True,
            type="primary"
        )
    
    if predict_button:
        # Prepare input data
        features_array = np.array([[
            alcohol,
            malic_acid,
            total_phenols,
            flavanoids,
            color_intensity,
            proline
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].numpy()
            prediction = torch.argmax(outputs, dim=1).item()
        
        confidence = probabilities[prediction]
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        # Main prediction
        col_result1, col_result2 = st.columns([2, 1])
        
        with col_result1:
            if prediction == 0:
                cultivar_name = "🍇 Cultivar 0 (Class A)"
                color = "🟢"
            elif prediction == 1:
                cultivar_name = "🍷 Cultivar 1 (Class B)"
                color = "🔵"
            else:
                cultivar_name = "🍺 Cultivar 2 (Class C)"
                color = "🟡"
            
            st.success(f"## {color} Predicted: {cultivar_name}")
        
        with col_result2:
            st.metric(
                "Confidence Score",
                f"{confidence*100:.2f}%",
                delta=f"{confidence:.4f}"
            )
        
        # Detailed probability distribution
        st.markdown("#### 📈 Prediction Probabilities")
        
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        
        with prob_col1:
            st.metric(
                "Cultivar 0",
                f"{probabilities[0]*100:.2f}%",
                help="Probability for Class A"
            )
        
        with prob_col2:
            st.metric(
                "Cultivar 1",
                f"{probabilities[1]*100:.2f}%",
                help="Probability for Class B"
            )
        
        with prob_col3:
            st.metric(
                "Cultivar 2",
                f"{probabilities[2]*100:.2f}%",
                help="Probability for Class C"
            )
        
        # Probability bar chart
        prob_data = {
            'Cultivar 0': probabilities[0],
            'Cultivar 1': probabilities[1],
            'Cultivar 2': probabilities[2]
        }
        
        st.bar_chart(prob_data, height=300)
        
        # Input summary
        st.markdown("---")
        st.markdown("#### 🔍 Input Features Used")
        
        input_summary = {
            'Alcohol (%)': f"{alcohol:.1f}",
            'Malic Acid': f"{malic_acid:.1f}",
            'Total Phenols': f"{total_phenols:.1f}",
            'Flavanoids': f"{flavanoids:.1f}",
            'Color Intensity': f"{color_intensity:.1f}",
            'Proline': f"{proline:.0f}"
        }
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write("**Feature Values:**")
            for feature, value in list(input_summary.items())[:3]:
                st.write(f"  • {feature}: {value}")
        
        with summary_col2:
            st.write("**Feature Values (continued):**")
            for feature, value in list(input_summary.items())[3:]:
                st.write(f"  • {feature}: {value}")
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### ℹ️ About This System")
        st.info("""
        **Wine Cultivar Prediction Model**
        
        - **Framework**: PyTorch Neural Network
        - **Architecture**: 
          - Input: 6 features
          - Hidden 1: 64 neurons + BatchNorm
          - Hidden 2: 32 neurons + BatchNorm
          - Output: 3 cultivars
        - **Features**: 6 selected chemical properties
        - **Classes**: 3 wine cultivars
        - **Model Persistence**: PyTorch (.pth)
        
        **Performance Metrics**:
        - Accuracy: ~97%
        - Precision: ~97%
        - Recall: ~97%
        - F1-Score: ~97%
        """)
        
        st.markdown("---")
        st.markdown("### 🎓 Feature Information")
        st.write("""
        **Alcohol**: Wine's alcohol content by volume
        
        **Malic Acid**: Natural acid found in wine, affects tartness
        
        **Total Phenols**: Antioxidant compounds affecting color and taste
        
        **Flavanoids**: Phenolic compounds important for wine quality
        
        **Color Intensity**: Wine's color depth and saturation
        
        **Proline**: Amino acid affecting wine characteristics
        """)

else:
    st.error("""
    ❌ **Model Not Found!**
    
    Please run the model training first:
    ```bash
    python train_model.py
    ```
    
    This will:
    1. Load the wine dataset
    2. Preprocess and scale features
    3. Train the PyTorch neural network
    4. Save the model and scaler
    """)
