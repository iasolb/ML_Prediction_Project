import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Your Project Title")
st.markdown("Brief description of what this app predicts.")

feature_1 = st.sidebar.slider("Feature 1", min_value=0.0, max_value=100.0, value=50.0)
feature_2 = st.sidebar.selectbox("Feature 2", ["Option A", "Option B", "Option C"])

model = joblib.load("model.pkl")  # store results in pickle

input_data = pd.DataFrame({"feature_1": [feature_1], "feature_2": [feature_2]})
prediction = model.predict(input_data)[0]

st.metric("Prediction", f"{prediction:.2f}")
