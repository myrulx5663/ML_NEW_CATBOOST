import streamlit as st
import pandas as pd
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    return df

df = load_data()

# Sidebar - Title & Info
st.sidebar.title("💧 Water Potability Predictor")
st.sidebar.info("Predict whether water is safe for drinking based on chemical properties.")

# Main Title
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>💧 Water Potability Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Use the sliders to enter water quality parameters.</p>", unsafe_allow_html=True)

# Divider
st.markdown("---")

# Show Raw Data Option
if st.checkbox("Show Raw Data"):
    st.subheader("📊 Raw Water Quality Data")
    st.dataframe(df.style.set_properties(**{'background-color': '#f2f2f2'}).highlight_max(axis=0))

# Input Section
st.markdown("## 🔧 Enter Water Parameters:")

# Helper function to create sliders
def user_input_features():
    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.slider("pH", float(df['ph'].min()), float(df['ph'].max()), float(df['ph'].mean()))
        hardness = st.slider("Hardness", float(df['Hardness'].min()), float(df['Hardness'].max()), float(df['Hardness'].mean()))
        solids = st.slider("Solids", float(df['Solids'].min()), float(df['Solids'].max()), float(df['Solids'].mean()))
        chloramines = st.slider("Chloramines", float(df['Chloramines'].min()), float(df['Chloramines'].max()), float(df['Chloramines'].mean()))

    with col2:
        sulfate = st.slider("Sulfate", float(df['Sulfate'].min()), float(df['Sulfate'].max()), float(df['Sulfate'].mean()))
        conductivity = st.slider("Conductivity", float(df['Conductivity'].min()), float(df['Conductivity'].max()), float(df['Conductivity'].mean()))
        organic_carbon = st.slider("Organic Carbon", float(df['Organic_carbon'].min()), float(df['Organic_carbon'].max()), float(df['Organic_carbon'].mean()))

    with col3:
        trihalomethanes = st.slider("Trihalomethanes", float(df['Trihalomethanes'].min()), float(df['Trihalomethanes'].max()), float(df['Trihalomethanes'].mean()))
        turbidity = st.slider("Turbidity", float(df['Turbidity'].min()), float(df['Turbidity'].max()), float(df['Turbidity'].mean()))

    data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Fake Prediction (Replace with real model prediction)
np.random.seed(42)
prediction = np.random.choice([0, 1])
prediction_proba = [round(np.random.uniform(0.5, 1), 2) if prediction == 0 else round(np.random.uniform(0, 0.5), 2),
                    round(1 - np.random.uniform(0, 0.5), 2) if prediction == 1 else round(1 - np.random.uniform(0.5, 1), 2)]

# Display Prediction Result
st.markdown("## 🎯 Prediction Result")

if prediction == 1:
    st.markdown("<h3 style='color: #4CAF50;'>✅ The water is Potable</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='color: #E53935;'>❌ The water is Not Potable</h3>", unsafe_allow_html=True)

# Metric Box for Probability
prob_percent = prediction_proba[1] * 100
st.metric(label="Likelihood of Being Potable", value=f"{prob_percent:.2f}%")

# Progress Bar Visualization
progress_html = f"""
<div style="width: 100%; background-color: #e0e0e0; border-radius: 8px;">
  <div style="width: {prob_percent:.2f}%; background-color: #4CAF50; padding: 8px 0; text-align: center; border-radius: 8px; color: white; font-weight: bold;">
    {prob_percent:.2f}%
  </div>
</div>
"""

st.markdown(progress_html, unsafe_allow_html=True)

# Divider
st.markdown("---")
