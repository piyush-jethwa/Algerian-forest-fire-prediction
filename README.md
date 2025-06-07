# 🔥 Algerian Forest Fire Prediction

This project uses machine learning to predict forest fire occurrences in Algeria based on meteorological and fire weather indices data. The dataset contains observations from two regions — Bejaia and Sidi Bel-Abbes — collected during the summer of 2012.

---

## 📂 Dataset Overview

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/547/algerian%2Bforest%2Bfires%2Bdataset)
- **Regions Covered**:
  - Bejaia (Northeast Algeria)
  - Sidi Bel-Abbes (Northwest Algeria)
- **Time Frame**: June to September 2012
- **Total Instances**: 244 (122 per region)
- **Features**:
  - **Meteorological Data**: Temperature (°C), Relative Humidity (%), Wind Speed (km/h), Rain (mm)
  - **Fire Weather Indices**:
    - FFMC: Fine Fuel Moisture Code
    - DMC: Duff Moisture Code
    - DC: Drought Code
    - ISI: Initial Spread Index
    - BUI: Build-Up Index
    - FWI: Fire Weather Index
  - **Target**: `Classes` → `fire` or `not fire`

---

## 🧠 Objectives

- Analyze environmental factors that contribute to forest fires.
- Develop predictive models to classify fire occurrences.
- Deploy an interactive web application for real-time prediction.

---

## 🛠️ Project Structure

Algerian-forest-fire-prediction/
├── .devcontainer/ # Development container configuration
├── models/ # Trained ML model files
├── notebook/ # EDA and model development notebooks
├── streamlit_app.py # Streamlit app script
├── requirements.txt # Dependencies
└── README.md # Project documentation



---

## 📊 Exploratory Data Analysis (EDA)

- Handled null and inconsistent values.
- Visualized feature distributions.
- Correlation analysis to find key predictors.
- Class imbalance handled using stratified methods.

---

## 🤖 ML Models Used

- Logistic Regression
- Lasso Regression And Lasso CV
- Ridge Regression and Ridge CV


> Evaluation Metrics: Accuracy, Precision, Recall, F1-score

---

## 🚀 Streamlit App

An interactive app built using [Streamlit](https://streamlit.io/).

### 🔧 How to Run Locally:

```bash
git clone https://github.com/piyush-jethwa/Algerian-forest-fire-prediction.git
cd Algerian-forest-fire-prediction
pip install -r requirements.txt
streamlit run streamlit_app.py
