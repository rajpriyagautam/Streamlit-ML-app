# app.py

import joblib
import numpy as np
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("model/model.joblib")


model = load_model()

# -----------------------------
# App UI
# -----------------------------

st.title("🤖 Machine Learning Prediction App")

st.write(
    "Enter the feature values below and get a real-time prediction "
    "from the trained machine-learning model."
)

st.divider()

st.subheader("Input Features")

feature_1 = st.slider(
    "Feature 1",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

feature_2 = st.slider(
    "Feature 2",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

feature_3 = st.slider(
    "Feature 3",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

feature_4 = st.slider(
    "Feature 4",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

# -----------------------------
# Prediction
# -----------------------------

input_data = np.array([
    [feature_1, feature_2, feature_3, feature_4]
])

prediction = model.predict(input_data)[0]

st.divider()

st.subheader("Prediction")

st.success(f"Predicted class: **{prediction}**")

# Probability display when supported
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction Confidence")

    for label, probability in zip(model.classes_, probabilities):
        st.write(f"**{label}**: {probability:.2%}")
        st.progress(float(probability))