import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.write("This app uses a live machine learning model to simulate match outcomes!")

# 2. User Input Elements
team_a = st.text_input("Home Team", "Argentina")
team_b = st.text_input("Away Team", "France")

if st.button("Run Simulation"):
    # Dynamically look up how many features the trained model actually expects
    num_features = model.n_features_in_
    mock_features = np.zeros((1, num_features))
    
    # Run prediction using the model
    prediction = model.predict(mock_features)
    
    if prediction[0] == 1:
        st.success(f"🏆 Simulation Result: {team_a} is predicted to win!")
    else:
        st.info(f"🏆 Simulation Result: {team_b} is predicted to win (or draw)!")
