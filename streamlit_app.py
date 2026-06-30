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
    # Note: Because your model expects 135 features (Elo, rolling forms, etc.),
    # we pass a clean array of default values matching X_test's shape.
    # In a full app, you would look up the team's real current Elo here!
    mock_features = np.zeros((1, 135)) 
    
    # Simple placeholder logic to set Elo difference for the demonstration
    mock_features[0, 2] = 50.0  # elo_difference placeholder
    
    prediction = model.predict(mock_features)
    
    if prediction[0] == 1:
        st.success(f"🏆 Simulation Result: {team_a} is predicted to win!")
    else:
        st.info(f"🏆 Simulation Result: {team_b} is predicted to win (or draw)!")
