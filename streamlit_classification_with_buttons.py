import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

df  = pd.read_csv('dataset\inventory_data_noisy.csv')

df.head()

df.shape

## Checking Missing Values
df.isna().sum()

df['Quantity'].fillna(df['Quantity'].mean(),inplace=True)
df['Consumption'].fillna(df['Consumption'].mean(),inplace=True)
df.isna().sum()

## Handling missing  values in Categorical values (most frequent value)
df['Location'].fillna(df['Location'].mode()[0],inplace=True)
df.isna().sum()


df.duplicated().sum()

## drop the duplicated values
df.drop_duplicates(inplace=True)
df.duplicated().sum()

df.info()

df.nunique()

df.describe()

import streamlit as st

if st.button('Show Plot'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    df_eda = df
    
    # Setting up the plotting area
    plt.figure(figsize=(18, 12))
    
    
    ## Distribution of numerical columns
    plt.subplot(2, 3, 1)
    sns.histplot(df_eda['Quantity'], kde=True, color='blue')
    plt.title('Distribution of Quantity')
    
    plt.subplot(2, 3, 2)
    sns.histplot(df_eda['Consumption'], kde=True, color='blue')
    plt.title('Distribution of Consumption')

import streamlit as st

if st.button('Show Plot'):
    ## Boxplots to visualize outliers
    plt.subplot(2, 3, 3)
    sns.boxplot(data=df_eda, x='Quantity', color='blue')
    plt.title('Boxplot of Quantity')
    
    plt.subplot(2, 3, 4)
    sns.boxplot(data=df_eda, x='Consumption', color='blue')
    plt.title('Boxplot of Consumption')

# Function to remove outliers using IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)  # First quartile (25%)
        Q3 = df[col].quantile(0.75)  # Third quartile (75%)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

# Select the columns for outlier removal (adjust as needed)
columns_to_filter = ["Quantity", "Consumption"]  
df= remove_outliers_iqr(df, columns_to_filter)




import streamlit as st

if st.button('Show Plot'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    df_eda = df
    
    # Setting up the plotting area
    plt.figure(figsize=(18, 12))
    
    
    ## Distribution of numerical columns
    plt.subplot(2, 3, 1)
    sns.histplot(df_eda['Quantity'], kde=True, color='blue')
    plt.title('Distribution of Quantity')
    
    plt.subplot(2, 3, 2)
    sns.histplot(df_eda['Consumption'], kde=True, color='blue')
    plt.title('Distribution of Consumption')

import streamlit as st

if st.button('Show Plot'):
    ## Boxplots to visualize outliers
    plt.subplot(2, 3, 3)
    sns.boxplot(data=df, x='Quantity', color='blue')
    plt.title('Boxplot of Quantity')
    
    plt.subplot(2, 3, 4)
    sns.boxplot(data=df, x='Consumption', color='blue')
    plt.title('Boxplot of Consumption')

print("Categories in 'Dealer_ID' variable:     ",end=" " )
print(df['Dealer_ID'].unique())

print("Categories in 'Product_ID' variable:  ",end=" ")
print(df['Product_ID'].unique())

print("Categories in'Location' variable:",end=" " )
print(df['Location'].unique())


# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))

## Adding column requirement based on Quantity and Consumption
low_threshold = df["Consumption"].quantile(0.33)
high_threshold = df["Consumption"].quantile(0.66)

# Create a new categorical target column
def categorize_consumption(value):
    if value <= low_threshold:
        return "Low"
    elif value <= high_threshold:
        return "Medium"
    else:
        return "High"

df["Consumption_Level"] = df["Consumption"].apply(categorize_consumption)

df['Quantity_consumption_ratios'] = df['Quantity']/df['Consumption']

df.head()

import streamlit as st

if st.button('Show Plot'):
    ## Histogram & KDE
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    plt.subplot(121)
    sns.histplot(data=df,x='Quantity',bins=30,kde=True,color='g')
    plt.subplot(122)
    sns.histplot(data=df,x='Quantity',kde=True,hue='Consumption_Level')
    plt.show()

import streamlit as st

if st.button('Show Plot'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    plt.subplot(121)
    sns.histplot(data=df,x='Consumption',bins=30,kde=True,color='g')
    plt.subplot(122)
    sns.histplot(data=df,x='Consumption',kde=True,hue='Consumption_Level')
    plt.show()

import streamlit as st

if st.button('Show Plot'):
    plt.subplots(1,3,figsize=(25,6))
    plt.subplot(141)
    sns.histplot(data=df,x='Quantity',kde=True,hue='Consumption_Level')
    plt.subplot(142)
    sns.histplot(data=df[df.Location=='Chicago'],x='Quantity',kde=True,hue='Consumption_Level')
    plt.subplot(143)
    sns.histplot(data=df[df.Location=='Los Angeles'],x='Quantity',kde=True,hue='Consumption_Level')
    plt.subplot(144)
    sns.histplot(data=df[df.Location=='New York'],x='Quantity',kde=True,hue='Consumption_Level')
    
    plt.show()

import streamlit as st

if st.button('Show Plot'):
    ## Multivariate analysis using pieplot
    
    plt.rcParams['figure.figsize'] = (30, 12)
    
    plt.subplot(1, 5, 1)
    size = df['Location'].value_counts()
    labels = 'Chicago', 'Los Angeles','New York'
    color = ['red','green','blue']
    
    
    plt.pie(size, colors = color, labels = labels,autopct = '.%2f%%')
    plt.title('Location', fontsize = 20)
    plt.axis('off')
    
    
    
    plt.subplot(1, 5, 2)
    size = df['Dealer_ID'].value_counts()
    labels = 'D000', 'D001','D002','D003','D004','D005','D006','D007','D008','D009'
    color = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
    
    plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
    plt.title('Dealer_ID', fontsize = 20)
    plt.axis('off')
    
    
    
    plt.subplot(1, 5, 3)
    size = df['Product_ID'].value_counts()
    labels = 'P000', 'P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010', 'P011', 'P012', 'P013', 'P014', 'P015', 'P016', 'P017', 'P018', 'P019'
    color = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
    
    plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
    plt.title('Product_ID', fontsize = 20)
    plt.axis('off')
    
    
    plt.subplot(1, 5, 4)
    size = df['Consumption_Level'].value_counts()
    labels = 'Low', 'Medium','High'
    color = ['red','green','blue']
    
    plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
    plt.title('Consumption Level', fontsize = 20)
    plt.axis('off')
    
    
    plt.tight_layout()
    plt.grid()
    
    plt.show()

import streamlit as st

if st.button('Show Plot'):
    f,ax=plt.subplots(1,2,figsize=(20,10))
    sns.countplot(x=df['Location'],data=df,palette ='bright',ax=ax[0],saturation=0.95)
    for container in ax[0].containers:
        ax[0].bar_label(container,color='black',size=20)
        
    plt.pie(x=df['Location'].value_counts(),labels=['Chicago','Los Angeles','New York'],explode=[0,0.1,0],autopct='%1.1f%%',shadow=True,colors=['#ff4d4d','#ff8000','#ff6000'])
    plt.show()

location_group = df.groupby('Location')
location_group

import streamlit as st

if st.button('Show Plot'):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Grouping by 'Location' and aggregating using the sum of the relevant columns
    location_group = df.groupby('Location').sum()
    # Extracting the required data from the aggregated DataFrame
    chicago_scores = [
        location_group.loc['Chicago', 'Consumption'], 
        location_group.loc['Chicago', 'Quantity']
    ]
    losangeles_scores = [
        location_group.loc['Los Angeles', 'Consumption'], 
        location_group.loc['Los Angeles', 'Quantity']
    ]
    newyork_scores = [
        location_group.loc['New York', 'Consumption'], 
        location_group.loc['New York', 'Quantity']
    ]
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    X = ['Consumption Average', 'Quantity Average']
    X_axis = np.arange(len(X))
    
    # Plot the bars with proper offsets for each city
    plt.bar(X_axis - 0.2, chicago_scores, 0.2, label='Chicago')
    plt.bar(X_axis, losangeles_scores, 0.2, label='Los Angeles')
    plt.bar(X_axis + 0.2, newyork_scores, 0.2, label='New York')
    
    # Set the x-axis labels and the chart's title
    plt.xticks(X_axis, X)
    plt.ylabel("Scores")
    plt.title(" Consumption Ratios vs Quantity", fontweight='bold')
    
    # Show legend and plot
    plt.legend()
    plt.show()
    

import streamlit as st

if st.button('Show Plot'):
    f,ax=plt.subplots(1,2,figsize=(20,10))
    sns.countplot(x=df['Dealer_ID'],data=df,palette = 'bright',ax=ax[0],saturation=0.95)
    for container in ax[0].containers:
        ax[0].bar_label(container,color='black',size=20)
        
    plt.pie(x = df['Dealer_ID'].value_counts(),labels=df['Dealer_ID'].value_counts().index,explode=[0.1,0,0,0,0,0,0,0,0,0],autopct='%1.1f%%',shadow=True)
    plt.show() 

import streamlit as st

if st.button('Show Plot'):
    Group_data2=df.groupby('Dealer_ID')
    f,ax=plt.subplots(1,2,figsize=(20,8))
    sns.barplot(x=Group_data2['Quantity'].mean().index,y=Group_data2['Quantity'].mean().values,palette = 'mako',ax=ax[0])
    ax[0].set_title('Quantity',color='#005ce6',size=20)
    
    for container in ax[0].containers:
        ax[0].bar_label(container,color='black',size=15)
    
    sns.barplot(x=Group_data2['Consumption'].mean().index,y=Group_data2['Consumption'].mean().values,palette = 'flare',ax=ax[1])
    ax[1].set_title('Consumption',color='#005ce6',size=20)
    
    for container in ax[1].containers:
        ax[1].bar_label(container,color='black',size=15)
    

import streamlit as st

if st.button('Show Plot'):
    plt.rcParams['figure.figsize'] = (15, 9)
    plt.style.use('fivethirtyeight')
    sns.countplot(df['Product_ID'], palette = 'Blues')
    plt.title('Comparison of Product ID', fontweight = 30, fontsize = 20)
    plt.xlabel('P000')
    plt.ylabel('count')
    plt.show()

import streamlit as st

if st.button('Show Plot'):
    df.groupby('Product_ID').agg('sum').plot(kind='barh',figsize=(10,10))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

df.head()

## Performing label encoding for columns

import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


df['Dealer_ID'] = label_encoder.fit_transform(df['Dealer_ID'])
df['Product_ID'] = label_encoder.fit_transform(df['Product_ID'])
df['Location'] = label_encoder.fit_transform(df['Location'])
df['Consumption_Level'] = label_encoder.fit_transform(df['Consumption_Level'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df['Quantity'] = scaler.fit_transform(df[['Quantity']])  # Use double brackets to pass as DataFrame
df['Consumption'] = scaler.fit_transform(df[['Consumption']])


df = df.drop('Date',axis=1)


## 1. Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X= df.drop(['Consumption_Level','Quantity_consumption_ratios'],axis=1)
y= df['Consumption_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)
clf.save()
acc = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Predictions
y_pred = clf.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report (Precision, Recall, F1-score)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC Score (Only for binary classification)
if len(set(y)) == 2:  # Check if binary classification
    auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"\nROC-AUC Score: {auc_score:.2f}")

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Predictions
y_pred = clf.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report (Precision, Recall, F1-score)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC Score (Only for binary classification)
if len(set(y)) == 2:  # Check if binary classification
    auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"\nROC-AUC Score: {auc_score:.2f}")



import joblib

# Save the trained LabelEncoder
joblib.dump(label_encoder, "label_encoder.pkl")

# Save the trained StandardScaler
joblib.dump(scaler, "scaler.pkl")

# Save the trained Logistic Regression model
joblib.dump(clf, "logistic_model.pkl")

# Save the feature names used during training
joblib.dump(X_train.columns.tolist(), "X_train_columns.pkl")

print("All objects saved successfully!")


# New sample data (replace with actual values)
new_data = pd.DataFrame({
    'Dealer_ID': ['D003'],
    'Product_ID': ['P010'],
     'Quantity': [50],
    'Consumption': [9], 
    'Location': ['Chicago']
   
})

# Apply Label Encoding (using trained encoder)
for col in ['Dealer_ID', 'Product_ID', 'Location']:
    if new_data[col][0] in label_encoder.classes_:
        new_data[col] = label_encoder.fit_transform(new_data[col])
    else:
        new_data[col] = -1  # Handle unseen labels

# Apply Scaling (using trained scaler)
new_data['Quantity'] = scaler.fit_transform(new_data[['Quantity']])  # Use double brackets to pass as DataFrame
new_data['Consumption'] = scaler.fit_transform(new_data[['Consumption']])

X# Reorder columns in X_new to match X_train
X_new = new_data[X_train.columns]  # This ensures the order matches

# Convert to NumPy array before scaling (prevents feature mismatch errors)
# Convert back to DataFrame with correct column names
X_new_scaled = pd.DataFrame(X_new, columns=X_train.columns)

# Predict Consumption Level
predicted_class = clf.predict(X_new_scaled)
predicted_proba = clf.predict_proba(X_new_scaled)

print(f"Predicted Consumption Level: {predicted_class[0]}")
print(f"Prediction Probabilities: {predicted_proba}")


