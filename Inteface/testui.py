import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Sidebar menu
menu = st.sidebar.radio("### Navigation", ["Home", "Data Upload", "Predictions", "Model Metrics"])

if menu == "Home":
    st.title("Welcome to the Dealer Consumption Prediction Model")
    st.image(r"C:\Users\Vamshi\Downloads\Customer-Segmentation.png", caption="Consumer behaviour and prediction")
    st.write("You can add your abstract over here")
elif menu == "Data Upload":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.text("Performing EDA on the above File")
        st.write("### Dataset Preview")
        # df.drop(columns='Unnamed:0',axis=1)
        st.dataframe(df.head())


        st.header("Exploratory Data Analysis")
        all_columns = df.columns.tolist()

        # Summary Statistics
        if st.checkbox("Show Summary Statistics"):
            st.write("### Summary Statistics")
            st.write(df.describe())
    
        # Correlation Heatmap
        if st.checkbox("Show Correlation Heatmap"):
            st.write("### Correlation Heatmap")
            corr_matrix = df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)  # Explicitly passing the figure

        if st.checkbox("Visualizations"):


            selected_column = st.sidebar.selectbox("Select column for histogram", all_columns)
    
            if selected_column:
                st.write(f"### Histogram of {selected_column}")
                fig, ax = plt.subplots()
                sns.histplot(df[selected_column], bins=20, kde=True, ax=ax)
                st.pyplot(fig)  # Explicitly passing the figure
    
                # Select features for scatter plot
                st.sidebar.write("#### Scatter Plot axis specifiers")
                selected_x_col = st.sidebar.selectbox(" Select X-axis for scatter plot", all_columns)
                selected_y_col = st.sidebar.selectbox("Select Y-axis for scatter plot", all_columns, index=1)
    
            if selected_x_col and selected_y_col:
                st.write(f"### Scatter Plot between {selected_x_col} and {selected_y_col}")
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=selected_x_col, y=selected_y_col, ax=ax)
                st.pyplot(fig)  # Explicitly passing the figure

        if st.checkbox("Filter Rows"):
            st.sidebar.write("#### Scatter Plot axis specifiers")
            filter_column = st.sidebar.selectbox("Select column for filtering", all_columns)
            unique_values = df[filter_column].unique()
    
            selected_value = st.sidebar.selectbox(f"Select value for {filter_column}", unique_values)
    
            if selected_value:
                filtered_data = df[df[filter_column] == selected_value]
                st.write(f"### Filtered Data by {filter_column} = {selected_value}")
                st.dataframe(filtered_data)

        # Multivariate Analysis Section
        st.header("Multivariate Analysis")
    
        # Pairplot (relationship between all numeric columns)
        if st.checkbox("Show Pairplot (multivariate analysis)"):
            st.write("### Pairplot of Numeric Columns")
            fig = sns.pairplot(df)  # Create pairplot
            st.pyplot(fig)
    
        # Jointplot (relationship between two columns)
        if st.checkbox("Joint plot for two columns"):
            st.sidebar.write("#### Joint Plot axis specifiers")
            joint_x_col = st.sidebar.selectbox("Select X-axis for joint plot", all_columns)
            joint_y_col = st.sidebar.selectbox("Select Y-axis for joint plot", all_columns, index=1)
    
            if joint_x_col and joint_y_col:
                st.write(f"### Joint Plot between {joint_x_col} and {joint_y_col}")
                fig = sns.jointplot(x=joint_x_col, y=joint_y_col, data=df, kind='hex')  # Kind can be 'scatter', 'kde', 'hex', 'reg', etc.
                st.pyplot(fig)
    
        # KDE Plot (Kernel Density Estimation for two variables)
        if st.checkbox("KDE Plot"):
            kde_x_col = st.sidebar.selectbox("Select X-axis for KDE plot", all_columns, index=0)
            kde_y_col = st.sidebar.selectbox("Select Y-axis for KDE plot", all_columns, index=1)
    
            if kde_x_col and kde_y_col:
                st.write(f"### KDE Plot between {kde_x_col} and {kde_y_col}")
                fig, ax = plt.subplots()
                sns.kdeplot(data=df, x=kde_x_col, y=kde_y_col, ax=ax, cmap="Blues", fill=True)
                st.pyplot(fig)





elif menu == "Predictions":
    st.title("Consumption Level Prediction")
    # Load trained models
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    clf = joblib.load("logistic_model.pkl")
    X_train_columns = joblib.load("X_train_columns.pkl")  # Save X_train.columns separately


    # Sidebar: Model Input Section
    st.header("Model Input Section")
    # Dealer ID Dropdown
    dealer_id = st.selectbox("Dealer ID", ["D001", "D002", "D003", "D004", "D005", "D006", "D007", "D008", "D009"])
    # Product ID Dropdown
    product_id = st.selectbox("Product ID", ["P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008", "P009", "P010", 
                                                  "P011", "P012", "P013", "P014", "P015", "P016", "P017", "P018", "P019"])
    # Quantity Slider (0 to 1000)
    quantity = st.slider("Quantity", 0, 1000, 500)  # Default at 500
    # Consumption Slider (0 to 100)
    consumption = st.slider("Consumption", 0, 100, 50)  # Default at 50
    # Location Dropdown
    location = st.selectbox("Location", ["New York", "Chicago", "Los Angeles"])

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





elif menu == "Model Metrics":
    st.title("Model Performance")
    st.write("Accuracy: 95%")
    st.write("Precision: 93%")
    # Show other metrics
