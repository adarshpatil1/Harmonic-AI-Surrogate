import streamlit as st
import numpy as np
import joblib
import os
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Damped Oscillator", layout="centered")

# Model loader
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model/harmonic_surrogate_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_model()

if model is None:
    st.stop()

st.title("🔮 Predict Final Position of Damped Oscillator at t = 10s")

# Input sliders
x0 = st.sidebar.slider("Initial Position x₀", -10.0, 10.0, 0.0)
v0 = st.sidebar.slider("Initial Velocity v₀", -5.0, 5.0, 0.0)
m = st.sidebar.slider("Mass m", 0.5, 2.0, 1.0)
c = st.sidebar.slider("Damping c", 0.1, 1.0, 0.5)
k = st.sidebar.slider("Spring Constant k", 1.0, 5.0, 4.0)

# Predict using model
X = np.array([[x0, v0, m, c, k]])
X_scaled = scaler.transform(X)
pred = model.predict(X_scaled)[0]

st.subheader(f"📈 Predicted Final Position: {pred:.4f}")

# Optional: plot true simulation
def simulate(x0, v0, m, c, k):
    def dydt(t, y):
        return [y[1], -(c/m)*y[1] - (k/m)*y[0]]
    t = np.linspace(0, 10, 200)
    sol = solve_ivp(dydt, [0, 10], [x0, v0], t_eval=t)
    return sol.t, sol.y[0]

if st.checkbox("🔍 Show True Oscillation Plot"):
    t, x = simulate(x0, v0, m, c, k)
    fig, ax = plt.subplots()
    ax.plot(t, x, label="True Position x(t)")
    ax.axhline(pred, color='red', linestyle='--', label="Predicted x(10)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position")
    ax.set_title("Damped Oscillation")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

