# ==============================
# House Price Prediction System
# ==============================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# Load the trained model
# -------------------------------
data = joblib.load("model/house_price_model.pkl")
model = data["model"]
scaler = data.get("scaler", None)  # in case you scaled features
feature_columns = data["feature_columns"]

# -------------------------------
# App Title
# -------------------------------
st.title("🏠 House Price Prediction System")
st.write(
    "Enter the house details below to predict the house price."
)

# -------------------------------
# User Inputs
# -------------------------------
# Replace these with the 6 features you selected
overall_qual = st.selectbox("Overall Quality (OverallQual)", list(range(1, 11)), index=5)
gr_liv_area = st.number_input("Above Ground Living Area (GrLivArea in sq.ft.)", min_value=200.0, value=1500.0)
total_bsmt_sf = st.number_input("Total Basement Area (TotalBsmtSF in sq.ft.)", min_value=0.0, value=800.0)
garage_cars = st.number_input("Number of Garage Cars (GarageCars)", min_value=0, value=2)
bedroom_abvgr = st.number_input("Number of Bedrooms Above Ground (BedroomAbvGr)", min_value=0, value=3)
full_bath = st.number_input("Number of Full Bathrooms (FullBath)", min_value=0, value=2)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict Price"):
    # Arrange inputs in the SAME ORDER as used during training
    input_data = np.array([
        overall_qual,
        gr_liv_area,
        total_bsmt_sf,
        garage_cars,
        bedroom_abvgr,
        full_bath
    ]).reshape(1, -1)

    # Scale input if scaler exists
    if scaler:
        input_data = scaler.transform(input_data)

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    # Display result
    st.success(f"🏡 Predicted House Price: ${predicted_price:,.2f}")

# -------------------------------
# Disclaimer
# -------------------------------
st.markdown(
    """
    ---
    **Note:**  
    This system is developed strictly for educational purposes and should not be used for actual financial decisions.
    """
)
