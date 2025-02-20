import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load the trained model & encoders
model = joblib.load(r"C:\Users\Vamshi\Desktop\custommodels\CVRUsecases\InvenManagementSystem\final_notebooks\regression_model.pkl")
encoder = joblib.load(r"C:\Users\Vamshi\Desktop\custommodels\CVRUsecases\InvenManagementSystem\final_notebooks\re_label_encoder.pkl")
date_ref = joblib.load(r"C:\Users\Vamshi\Desktop\custommodels\CVRUsecases\InvenManagementSystem\final_notebooks\date_reference.pkl")

# Predefined Dealer & Product IDs
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

    new_data = pd.DataFrame({
    'Date': [input_date],  # New date
    'Dealer_ID': [selected_dealer],   # Sample Dealer ID
    'Product_ID': [selected_product],  # Sample Product ID
    'Quantity': [quantity]         # Sample Quantity
})

    # Convert Date to numerical format
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    new_data['Days'] = (new_data['Date'] - new_data['Date'].min()).dt.days  # Use the same reference date


    # Encode categorical values
    def encode_label(value, encoder):
        st.write("Dealer Encoder Classes: ", encoder.classes_)

        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            st.error(f"Error: The value '{value}' was not found in the encoder classes.")
            st.stop()

    new_data['Dealer_ID'] = new_data['Dealer_ID'].apply(lambda x: encode_label(x, encoder))
    new_data['Product_ID'] = new_data['Product_ID'].apply(lambda x: encode_label(x, encoder))

    X_new = new_data[['Days', 'Dealer_ID', 'Product_ID', 'Quantity']]

    # Predict consumption
    
    predicted_consumption = model.predict(X_new)
    st.success(f"🔮 **Predicted Consumption:** {predicted_consumption:.2f}")
    
