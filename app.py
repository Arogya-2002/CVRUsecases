import streamlit as st
import pandas as pd
import joblib  # To load trained models

# Load trained models
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
clf = joblib.load("logistic_model.pkl")
X_train_columns = joblib.load("X_train_columns.pkl")  # Save X_train.columns separately

# Streamlit UI
st.title("Consumption Level Prediction")

# User Input
dealer_id = st.text_input("Dealer ID (e.g., D003)")
product_id = st.text_input("Product ID (e.g., P010)")
location = st.text_input("Location (e.g., Chicago)")
quantity = st.number_input("Quantity", min_value=1, max_value=1000, value=50)
consumption = st.number_input("Consumption", min_value=1, max_value=100, value=9)

# Prediction Button
if st.button("Predict"):
    # Create DataFrame
    new_data = pd.DataFrame({
        'Dealer_ID': [dealer_id],
        'Product_ID': [product_id],
        'Quantity': [quantity],
        'Consumption': [consumption],
        'Location': [location]
    })

    # Apply Label Encoding (using trained encoder)
    for col in ['Dealer_ID', 'Product_ID', 'Location']:
        if new_data[col][0] in label_encoder.classes_:
            new_data[col] = label_encoder.fit_transform(new_data[col])
        else:
            new_data[col] = -1  # Handle unseen labels

    # Apply Scaling (using trained scaler)
    new_data[['Quantity', 'Consumption']] = scaler.fit_transform(new_data[['Quantity', 'Consumption']])

    # Ensure correct column order
    X_new = pd.DataFrame(columns=X_train_columns)
    X_new.loc[0] = new_data[X_train_columns].iloc[0]

    # Predict Consumption Level
    predicted_class = clf.predict(X_new)
    predicted_proba = clf.predict_proba(X_new)

    # Display Results
    st.success(f"Predicted Consumption Level: {predicted_class[0]}")
    st.write(f"Prediction Probabilities: {predicted_proba}")

