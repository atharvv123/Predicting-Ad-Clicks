import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("pred_ad_click.pkl")

# Title of the Streamlit App
st.title("Ad Click Prediction")

# Create input fields for user data
st.sidebar.header("User Input Features")

daily_time_spent = st.sidebar.number_input("Daily Time Spent on Site (minutes)", min_value=30.0, max_value=100.0, value=65.0, step=0.1)
age = st.sidebar.number_input("Age", min_value=18, max_value=80, value=35, step=1)
area_income = st.sidebar.number_input("Area Income (USD)", min_value=10000, max_value=100000, value=55000, step=500)
daily_internet_usage = st.sidebar.number_input("Daily Internet Usage (minutes)", min_value=50.0, max_value=300.0, value=180.0, step=0.1)
male = st.sidebar.radio("Gender", [0, 1], index=1, format_func=lambda x: "Male" if x == 1 else "Female")
hour = st.sidebar.slider("Hour of Interaction", min_value=0, max_value=23, value=12)
day = st.sidebar.slider("Day of Interaction", min_value=1, max_value=31, value=15)

# Convert input into a DataFrame
user_data = pd.DataFrame({
    "Daily Time Spent on Site": [daily_time_spent],
    "Age": [age],
    "Area Income": [area_income],
    "Daily Internet Usage": [daily_internet_usage],
    "Male": [male],
    "Hour": [hour],
    "Day": [day]
})

# Load the scaler and transform the input
scaler = StandardScaler()
user_data_scaled = scaler.fit_transform(user_data)  # Note: Ideally, use the scaler fitted on training data

# Make predictions
prediction = model.predict(user_data_scaled)
prediction_proba = model.predict_proba(user_data_scaled)[:, 1]

# Display Prediction
st.subheader("Prediction Result")
if prediction[0] == 1:
    st.success("User is likely to click on the advertisement!")
else:
    st.warning("User is unlikely to click on the advertisement.")

# Display Probability
st.subheader("Prediction Probability")
st.write(f"Probability of clicking the ad: {prediction_proba[0]:.2f}")

