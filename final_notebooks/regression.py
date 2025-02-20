import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "dataset\inventory_data_continuous.csv"
df = pd.read_csv(file_path)

# Convert 'Date' to numerical format (days since first date)
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Encode categorical variables
encoder = LabelEncoder()
df['Dealer_ID'] = encoder.fit_transform(df['Dealer_ID'])
df['Product_ID'] = encoder.fit_transform(df['Product_ID'])

# Select features (X) and target (y)
X = df[['Days', 'Dealer_ID', 'Product_ID', 'Quantity']]
y = df['Consumption']

# Handle missing values by filling with median
X = X.fillna(X.median())
y = y.fillna(y.median())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R² Score: {r2}')

# Sample new data (replace with actual values)
new_data = pd.DataFrame({
    'Date': ['2025-02-10'],  # New date
    'Dealer_ID': ['D003'],   # Sample Dealer ID
    'Product_ID': ['P002'],  # Sample Product ID
    'Quantity': [70]         # Sample Quantity
})



# Convert Date to numerical format
new_data['Date'] = pd.to_datetime(new_data['Date'])
new_data['Days'] = (new_data['Date'] - df['Date'].min()).dt.days  # Use the same reference date


# Handle unseen labels safely
def encode_label(label, encoder):
    if label in encoder.classes_:
        return encoder.transform([label])[0]  # Encode if known
    else:
        return -1

new_data['Dealer_ID'] = new_data['Dealer_ID'].apply(lambda x: encode_label(x, encoder))
new_data['Product_ID'] = new_data['Product_ID'].apply(lambda x: encode_label(x, encoder))

# Select relevant features
X_new = new_data[['Days', 'Dealer_ID', 'Product_ID', 'Quantity']]

# Predict consumption
predicted_consumption = model.predict(X_new)
print(f'Predicted Consumption: {predicted_consumption[0]}')

