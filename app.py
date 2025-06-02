import streamlit as st
import numpy as np
import joblib
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Damped Oscillator Predictor", layout="centered")

# Load model and scaler
model = joblib.load('harmonic_surrogate_model.pkl')
scaler = joblib.load('scaler.pkl')  # If you used StandardScaler

# App title
st.title("🔮 Damped Harmonic Oscillator Final Position Predictor")

# Sidebar for input parameters
st.sidebar.header("🛠 Input Parameters")

x0 = st.sidebar.slider("Initial Position (x₀)", -10.0, 10.0, 0.0)
v0 = st.sidebar.slider("Initial Velocity (v₀)", -5.0, 5.0, 0.0)
m = st.sidebar.slider("Mass (m)", 0.5, 2.0, 1.0)
c = st.sidebar.slider("Damping Coefficient (c)", 0.1, 1.0, 0.5)
k = st.sidebar.slider("Spring Constant (k)", 1.0, 5.0, 4.0)

# Format input and make prediction
input_features = np.array([[x0, v0, m, c, k]])
input_scaled = scaler.transform(input_features)  # Only if scaler was used

prediction = model.predict(input_scaled)[0]

# Show prediction
st.subheader("📈 Predicted Final Position at t = 10s")
st.success(f"Predicted x(10) = {prediction:.4f}")

# Optional: Simulate true trajectory
def simulate_oscillator(x0, v0, m, c, k, t_span=(0, 10), t_eval=None):
    def oscillator(t, y):
        return [y[1], -(c/m)*y[1] - (k/m)*y[0]]
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)
    sol = solve_ivp(oscillator, t_span, [x0, v0], t_eval=t_eval)
    return sol.t, sol.y[0]

# Checkbox for visualization
if st.checkbox("🔍 Show Full Oscillation Plot"):
    t, x = simulate_oscillator(x0, v0, m, c, k)
    fig, ax = plt.subplots()
    ax.plot(t, x, label="True x(t)", color='blue')
    ax.axhline(y=prediction, color='red', linestyle='--', label="Predicted x(10)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position")
    ax.set_title("Oscillator Behavior Over Time")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
