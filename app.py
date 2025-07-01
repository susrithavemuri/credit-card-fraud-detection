import streamlit as st
import pandas as pd
import joblib
from src.preprocessing import scale_features
from src.model import train_logistic_regression

st.title("Credit Card Fraud Detection 💳🔍")

uploaded_file = st.file_uploader("Upload CSV file with transaction data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Original Data:", df.head())

    df_processed = scale_features(df)
    st.write("Processed Data:", df_processed.head())

    # Dummy target to split X, y
    if 'Class' in df_processed.columns:
        X = df_processed.drop("Class", axis=1)
        y = df_processed['Class']
    else:
        X = df_processed

    model = joblib.load("model_lr.pkl")  # or model_ensemble.pkl

    y_pred = model.predict(X)

    df['Prediction'] = y_pred
    st.write("Predictions:", df[['Prediction']])

    st.success("✅ Prediction Complete")
