import app as st
import pandas as pd
import joblib

# Load the saved model
best_rf_model = joblib.load('notebooks/best_random_forest_model.pkl')

# Title of the Streamlit app
st.title("Random Forest Classifier - CSV Prediction Interface")

# Allow user to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    input_data = pd.read_csv(uploaded_file)
    
    # Display the uploaded data
    st.write("Uploaded Data:")
    st.dataframe(input_data)

    # Check if 'Predict' button is clicked
    if st.button('Predict'):
        # Ensure the input data is properly formatted (optional: adjust columns as needed)
        # Assuming the CSV has the same feature columns as the training data
        predictions = best_rf_model.predict(input_data)
        
        # Display predictions
        st.write("Predictions:")
        st.write(predictions)
        
        # Option to download predictions
        output = pd.DataFrame(predictions, columns=['Predictions'])
        csv = output.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )
