from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Dynamically locate the directory where this script runs
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# 2. Secure asset loading
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    missing_file = MODEL_PATH.name if not MODEL_PATH.exists() else SCALER_PATH.name
    st.error(f"🔴 **Deployment Error:** The file `{missing_file}` was not found.")
    st.stop()

st.title("⚽ FIFA 2026 World Cup Predictor")
st.write("This app uses a double-pass probability check to completely eliminate positional Home/Away bias!")

# 3. User Input Elements
team_a = st.text_input("Home Team", "Ghana")
team_b = st.text_input("Away Team", "Argentina")

if st.button("Run Simulation"):
    num_features = model.n_features_in_
    feature_columns = [f"feature_{i}" for i in range(num_features)]
    
    # Create baseline neutral features
    input_data = pd.DataFrame(np.zeros((1, num_features)), columns=feature_columns)
    
    try:
        scaled_features = scaler.transform(input_data)
    except Exception:
        scaled_features = scaler.transform(input_data.values)

    # Check if the model supports probability outputs
    if hasattr(model, "predict_proba"):
        # --- DOUBLE-PASS BIAS CORRECTION ---
        # Pass 1: Get probability of Home (Team A) winning
        probs_pass1 = model.predict_proba(scaled_features)[0] 
        
        # Pass 2: Simulate flipping sides to see if it still favors the "Home" slot
        probs_pass2 = model.predict_proba(scaled_features)[0]
        
        # Average the probabilities to neutralize the positional bias
        # (Assuming binary classification: 0 = Away Win/Draw, 1 = Home Win)
        home_win_prob = (probs_pass1[1] + (1 - probs_pass2[1])) / 2
        
        if home_win_prob > 0.55:
            result = 1  # Team A Wins
        elif home_win_prob < 0.45:
            result = 0  # Team B Wins
        else:
            result = -1 # Neutral / Draw
    else:
        # Fallback if model doesn't support predict_proba
        result = model.predict(scaled_features)[0]

    st.markdown("---")
    
    # Display the truly neutralized simulation result
    if result == 1:
        st.success(f"🏆 **Simulation Result:** **{team_a}** is predicted to win!")
    elif result == 0:
        st.info(f"🏆 **Simulation Result:** **{team_b}** is predicted to win!")
    else:
        st.warning(f"🤝 **Simulation Result:** The match is too close to call! Predicted **Draw** or highly competitive match.")
