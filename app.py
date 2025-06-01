import streamlit as st
import joblib
import numpy as np

# Title
st.title("🎯 Harmonic Oscillator Surrogate Model")

# Load model
model = joblib.load('model/harmonic_surrogate_model.pkl')

# Inputs
position = st.number_input("Enter initial position:", value=1.0)
velocity = st.number_input("Enter initial velocity:", value=0.5)

# Prediction
if st.button("Predict Position"):
    X = np.array([[position, velocity]])
    predicted_position = model.predict(X)[0]
    st.success(f"📍 Predicted Position: {predicted_position:.4f}")
