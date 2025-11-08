import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

st.set_page_config(page_title="Calorie Burn Predictor", layout="wide")
st.title("🏋️‍♀️ Calorie Burn Prediction App")

# Upload CSV files
calories_file = st.file_uploader("Upload calories.csv", type="csv")
exercise_file = st.file_uploader("Upload exercise.csv", type="csv")

if calories_file and exercise_file:
    # Load data
    calories = pd.read_csv(calories_file)
    exercise_data = pd.read_csv(exercise_file)
    calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

    st.subheader("📊 Data Preview")
    st.dataframe(calories_data.head())

    # Preprocessing
    calories_data.replace({'Gender': {'male': 1, 'female': 0}}, inplace=True)
    X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
    Y = calories_data['Calories']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Train model
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    mae = metrics.mean_absolute_error(Y_test, predictions)

    st.subheader("📈 Model Performance")
    st.write(f"Mean Absolute Error: `{mae:.2f}`")

    # Optional: show prediction samples
    st.subheader("🔍 Sample Predictions")
    sample_df = pd.DataFrame({'Actual': Y_test.values[:10], 'Predicted': predictions[:10]})
    st.dataframe(sample_df)

    # Optional: correlation heatmap
    st.subheader("📌 Feature Correlation Heatmap")
    correlation = calories_data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='Blues', fmt='.1f', ax=ax)
    st.pyplot(fig)

else:
    st.info("Please upload both `calories.csv` and `exercise.csv` to begin.")
