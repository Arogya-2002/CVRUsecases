import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Load the trained model & encoders
model = joblib.load("regression_model.pkl")
encoder = joblib.load("label_encoder.pkl")
date_ref = joblib.load("date_reference.pkl")

# Predefined Dealer & Product IDs (replace with actual values from training data)
dealer_ids = ['D001', 'D002', 'D003', 'D004']
product_ids = ['P001', 'P002', 'P003', 'P004']

# Streamlit UI
st.title("📊 Inventory Consumption Prediction")

st.subheader("🔽 Enter Details for Prediction")

# User Inputs
input_date = st.date_input("📅 Select Date")
selected_dealer = st.selectbox("🏬 Select Dealer ID", dealer_ids)
selected_product = st.selectbox("📦 Select Product ID", product_ids)
quantity = st.number_input("🔢 Enter Quantity", min_value=1, max_value=1000, value=50)

# Predict Button
if st.button("🔮 Predict Consumption"):
    # Convert input date to numerical format
    days_since_start = (pd.to_datetime(input_date) - date_ref).days

    # Encode categorical values
    def encode_label(value, encoder):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return -1  # Assign -1 for unseen values

    encoded_dealer = encode_label(selected_dealer, encoder)
    encoded_product = encode_label(selected_product, encoder)

    # Create DataFrame for prediction
    new_data = pd.DataFrame([[days_since_start, encoded_dealer, encoded_product, quantity]], 
                            columns=['Days', 'Dealer_ID', 'Product_ID', 'Quantity'])

    # Predict consumption
    predicted_consumption = model.predict(new_data)[0]

    # Display prediction
    st.success(f"🔮 **Predicted Consumption:** {predicted_consumption:.2f}")
