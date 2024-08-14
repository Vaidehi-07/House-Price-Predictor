import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#Load and preprocess the dataset 
df = pd.read_csv(r"C:\Users\Vaidehi Dhamnikar\Downloads\house_prices.csv.txt")
df = df.dropna()
df_encoded = pd.get_dummies(df, columns=['location'], drop_first=True)

# Features and target variable

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("House Price Prediction")

# User inputs
bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, step=1)
sqft_living = st.number_input("Square Footage of Living Area", min_value=0, max_value=10000, step=1)
sqft_lot = st.number_input("Square Footage of Lot", min_value=0, max_value=100000, step=1)
floors = st.number_input("Number of Floors", min_value=0, max_value=10, step=1)
location = st.selectbox("Location", options=df['location'].unique())

# Prepare input data
input_data = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'sqft_lot': [sqft_lot],
    'floors': [floors],
    'location': [location]
})

# Convert categorical feature and ensure all columns are present
input_data_encoded = pd.get_dummies(input_data, columns=['location'], drop_first=True)
input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)

# Scale the input features
input_data_scaled = scaler.transform(input_data_encoded)

# Prediction
if st.button("Predict"):
    predicted_price = model.predict(input_data_scaled)
    st.write(f"Predicted Price: ${predicted_price[0]:,.2f}")
