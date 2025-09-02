import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------------
# Safe and cached model loading
# ------------------------------
@st.cache_resource
def load_model():
    model_path = r"C:\Users\princ\Documents\nareshit\EDA_sessions\Telecom_churn_streamlit\telecom_churn_model.pkl"  # relative path in the same folder
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    try:
        model = joblib.load(model_path)
        return model
    except EOFError:
        st.error("The model file is corrupted. Please recreate it.")
        st.stop()

# Load model
model = load_model()
st.success("Model loaded successfully!")

# Features used in training
features = list(model.feature_names_in_)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Telecom Customer Churn Prediction 🚀")
st.write("Enter the customer details below:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=35)
no_of_days_subscribed = st.number_input("No. of Days Subscribed", min_value=0, value=200)
multi_screen = st.selectbox("Multi-screen (0=No, 1=Yes)", [0, 1])
weekly_mins_watched = st.number_input("Weekly Minutes Watched", min_value=0, value=300)
minimum_daily_mins = st.number_input("Minimum Daily Minutes", min_value=0, value=5)
maximum_daily_mins = st.number_input("Maximum Daily Minutes", min_value=0, value=60)
videos_watched = st.number_input("Videos Watched", min_value=0, value=50)
customer_support_calls = st.number_input("Customer Support Calls", min_value=0, value=2)

# Prediction button
if st.button("Predict Churn"):
    # Prepare input DataFrame
    new_data = pd.DataFrame([{
        "age": age,
        "no_of_days_subscribed": no_of_days_subscribed,
        "multi_screen": multi_screen,
        "weekly_mins_watched": weekly_mins_watched,
        "minimum_daily_mins": minimum_daily_mins,
        "maximum_daily_mins": maximum_daily_mins,
        "videos_watched": videos_watched,
        "customer_support_calls": customer_support_calls
    }])[features]  # ensure correct order

    # Predict
    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[:, 1][0]

    # Display result
    if prediction == 1:
        st.error(f"The customer is likely to churn! (Probability: {probability:.2f})")
    else:
        st.success(f"The customer is not likely to churn. (Probability: {probability:.2f})")
