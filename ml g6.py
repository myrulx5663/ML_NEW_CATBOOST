import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="💧 Water Potability Predictor", page_icon="💧")

# Title and description
st.title("💧 Water Potability Prediction")
st.markdown("""
Enter the water quality parameters below to determine if the water is safe to drink.
All values should be numeric.
""")

# Load model function with error handling
@st.cache_resource
def load_model():
    model_path = "best_model_svc_calibrated.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. "
                                "Please ensure it is uploaded in the correct location.")
    
    data = joblib.load(model_path)
    return data["pipeline"], data["threshold"]

try:
    pipeline, threshold = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Input fields
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, help="Measure of acidity/alkalinity")
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=150.0, help="Water hardness in mg/L")
solids = st.number_input("Solids (ppm)", min_value=0.0, value=20000.0, help="Total dissolved solids in ppm")
chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0, help="Amount of chloramines in ppm")
sulfate = st.number_input("Sulfate (ppm)", min_value=0.0, value=300.0, help="Sulfate content in ppm")
conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=400.0, help="Electrical conductivity in μS/cm")
organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=15.0, help="Organic carbon content in ppm")
trihalomethanes = st.number_input("Trihalomethanes (ppm)", min_value=0.0, value=60.0, help="THM content in ppm")
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=5.0, help="Water turbidity in NTU")

# Create input DataFrame
input_data = pd.DataFrame([{
    "ph": ph,
    "Hardness": hardness,
    "Solids": solids,
    "Chloramines": chloramines,
    "Sulfate": sulfate,
    "Conductivity": conductivity,
    "Organic_carbon": organic_carbon,
    "Trihalomethanes": trihalomethanes,
    "Turbidity": turbidity
}])

# Predict button
if st.button("Predict Potability"):
    try:
        prob = pipeline.predict_proba(input_data)[0][1]
        prediction = "✅ Safe to Drink" if prob >= threshold else "❌ Not Safe"

        st.subheader("Prediction Result:")
        st.markdown(f"### Water is **{prediction}**")
        st.write(f"Probability Score: **{prob:.2f}**")
        st.write(f"Decision Threshold: **{threshold:.2f}**")
        st.progress(float(prob))

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
