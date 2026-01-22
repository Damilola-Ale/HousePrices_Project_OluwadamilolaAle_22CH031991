import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# Load trained model
# -------------------------------
data = joblib.load("model/titanic_survival_model.pkl")

model = data["model"]
scaler = data["scaler"]
feature_columns = data["feature_columns"]  # Ensures exact order used during training

# -------------------------------
# App Title
# -------------------------------
st.title("🚢 Titanic Survival Prediction System")

st.write(
    "Enter passenger details below to predict whether the passenger survived the Titanic disaster."
)

# -------------------------------
# User Inputs
# -------------------------------
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)

sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical inputs exactly like during training
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict Survival"):
    # Arrange input in the same order as feature_columns
    input_df = pd.DataFrame([[
        pclass,
        age,
        fare,
        sex_male,
        embarked_Q,
        embarked_S
    ]], columns=feature_columns)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display result
    if prediction == 1:
        st.success("✅ Prediction: Survived")
    else:
        st.error("❌ Prediction: Did Not Survive")

# -------------------------------
# Disclaimer
# -------------------------------
st.markdown(
    """
    ---
    **Note:**  
    This system is developed strictly for educational purposes and should not be used for real-world decision making.
    """
)