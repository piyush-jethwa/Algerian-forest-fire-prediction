# streamlit_app.py

import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Streamlit UI
st.title("🔥 Algerian Forest Fire Prediction App")
st.write("Enter the input features below:")

# Input fields
Temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
RH = st.number_input("Relative Humidity (%)", min_value=0.0, step=0.1)
Ws = st.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
Rain = st.number_input("Rain (mm)", min_value=0.0, step=0.1)
FFMC = st.number_input("FFMC Index", min_value=0.0, step=0.1)
DMC = st.number_input("DMC Index", min_value=0.0, step=0.1)
ISI = st.number_input("ISI Index", min_value=0.0, step=0.1)
Classes = st.selectbox("Fire Class", options=[0, 1], format_func=lambda x: "No Fire" if x == 0 else "Fire")
Region = st.selectbox("Region", options=[1, 2], format_func=lambda x: f"Region {x}")

# Predict button
if st.button("Predict"):
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    scaled_data = standard_scaler.transform(input_data)
    result = ridge_model.predict(scaled_data)
    st.success(f"🔥 Predicted Value: {result[0]:.2f}")

