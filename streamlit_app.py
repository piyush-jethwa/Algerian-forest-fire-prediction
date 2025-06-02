# streamlit_app.py

import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
ridge_model = joblib.load('models/ridge.pkl')
standard_scaler = joblib.load('models/scaler.pkl')

# Streamlit UI
st.title("🔥 Algerian Forest Fire Prediction App")
st.write("Provide the meteorological and environmental data below to predict the forest fire index.")

# Input descriptions
st.markdown("### 🔍 Input Feature Descriptions:")
st.markdown("""
- **Temperature (°C):** Air temperature in degrees Celsius. (Typical range: 0 - 50°C)
- **Relative Humidity (%):** Amount of moisture in the air. (Range: 0 - 100%)
- **Wind Speed (km/h):** Wind speed in kilometers per hour. (Range: 0 - 100 km/h)
- **Rain (mm):** Rainfall in millimeters. (Range: 0 - 10 mm)
- **FFMC Index:** Fine Fuel Moisture Code, represents the moisture content of surface litter and influences fire ignition. (Range: 0 - 104)
- **DMC Index:** Duff Moisture Code, moisture content of decomposed organic material (duff). (Range: 0 - 200+)
- **ISI Index:** Initial Spread Index, expected rate of fire spread. (Range: 0 - 50+)
- **Fire Class:** 0 = No Fire, 1 = Fire (based on class labels in the dataset)
- **Region:** 1 = Bejaia Region, 2 = Sidi-Bel Abbes Region
""")

# Input fields
# Input fields with full forms and typical ranges in labels
Temperature = st.number_input("🌡️ Temperature (°C) (Typical range: 0 - 50°C)", min_value=0.0, max_value=50.0, step=0.1)
RH = st.number_input("💧 Relative Humidity (%) (Range: 0 - 100%)", min_value=0.0, max_value=100.0, step=0.1)
Ws = st.number_input("🌬️ Wind Speed (km/h) (Range: 0 - 100 km/h)", min_value=0.0, max_value=100.0, step=0.1)
Rain = st.number_input("☔ Rain (mm) (Range: 0 - 10 mm)", min_value=0.0, max_value=10.0, step=0.1)

FFMC = st.number_input("🔥 FFMC (Fine Fuel Moisture Code) Index (Range: 0 - 104)", min_value=0.0, max_value=104.0, step=0.1)
DMC = st.number_input("🌲 DMC (Duff Moisture Code) Index (Range: 0 - 300+)", min_value=0.0, max_value=300.0, step=0.1)
ISI = st.number_input("🚀 ISI (Initial Spread Index) (Range: 0 - 50+)", min_value=0.0, max_value=50.0, step=0.1)


Classes = st.selectbox("🔥 Fire Class (0 = No Fire, 1 = Fire)", options=[0, 1], format_func=lambda x: "No Fire (0)" if x == 0 else "Fire (1)")
Region = st.selectbox("📍 Region (1 = Bejaia, 2 = Sidi-Bel Abbes)", options=[1, 2], format_func=lambda x: "Region 1 (Bejaia)" if x == 1 else "Region 2 (Sidi-Bel Abbes)")

# Predict button
if st.button("Predict"):
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    scaled_data = standard_scaler.transform(input_data)
    result = ridge_model.predict(scaled_data)
    st.success(f"🔥 Predicted Fire Index Value: {result[0]:.2f}")
