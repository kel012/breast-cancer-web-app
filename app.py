import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model_top10.pkl", "rb"))
scaler = pickle.load(open("scaler_top10.pkl", "rb"))

# Title
st.title("Breast Cancer Web App")
st.write("Enter tumor details below to predict whether it is Benign or Malignant.")

# Inputs 
concave_points_worst = st.number_input("Concave Points Worst")
perimeter_worst = st.number_input("Perimeter Worst")
concave_points_mean = st.number_input("Concave Points Mean")
radius_worst = st.number_input("Radius Worst")
perimeter_mean = st.number_input("Perimeter Mean")
area_worst = st.number_input("Area Worst")
radius_mean = st.number_input("Radius Mean")
area_mean = st.number_input("Area Mean")
concavity_mean = st.number_input("Concavity Mean")
concavity_worst = st.number_input("Concavity Worst")

# Prediction
if st.button("Predict"):
    try:
        # Arrange in correct order
        features = np.array([[
            concave_points_worst,
            perimeter_worst,
            concave_points_mean,
            radius_worst,
            perimeter_mean,
            area_worst,
            radius_mean,
            area_mean,
            concavity_mean,
            concavity_worst
        ]])

        # Scale
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)

        # Output
        if prediction[0] == 1:
            st.error("Malignant Tumor")
        else:
            st.success("Benign Tumor")

        # Confidence
        st.write(f"Confidence: {probability[0][prediction[0]] * 100:.2f}%")

    except Exception as e:
        st.write("Error:", e)