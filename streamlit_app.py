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
team_a = st.text_input("Home Team", "Brazil")
team_b = st.text_input("Away Team", "Norway")

if st.button("Run Simulation"):
    num_features = model.n_features_in_
    feature_columns = [f"feature_{i}" for i in range(num_features)]
    
    # Generate mock distinct inputs so the model doesn't see identical data
    val_a = sum(ord(c) for c in team_a) % 10
    val_b = sum(ord(c) for c in team_b) % 10
    
    # Pass 1: Team A as Home, Team B as Away
    data_pass1 = np.zeros((1, num_features))
    if num_features > 1:
        data_pass1[0, 0] = val_a
        data_pass1[0, 1] = val_b
        
    # Pass 2: Flip sides completely (Team B as Home, Team A as Away)
    data_pass2 = np.zeros((1, num_features))
    if num_features > 1:
        data_pass2[0, 0] = val_b
        data_pass2[0, 1] = val_a

    input_p1 = pd.DataFrame(data_pass1, columns=feature_columns)
    input_p2 = pd.DataFrame(data_pass2, columns=feature_columns)
    
    try:
        scaled_p1 = scaler.transform(input_p1)
        scaled_p2 = scaler.transform(input_p2)
    except Exception:
        scaled_p1 = scaler.transform(input_p1.values)
        scaled_p2 = scaler.transform(input_p2.values)

    if hasattr(model, "predict_proba"):
        probs_pass1 = model.predict_proba(scaled_p1)[0] 
        probs_pass2 = model.predict_proba(scaled_p2)[0]
        
        # Calculate combined probability (closer to 1 favors Team A, closer to 0 favors Team B)
        home_win_prob = (probs_pass1[1] + (1 - probs_pass2[1])) / 2
        
        if home_win_prob > 0.52:
            result = 1
        elif home_win_prob < 0.48:
            result = 0
        else:
            result = -1
    else:
        pred1 = model.predict(scaled_p1)[0]
        pred2 = model.predict(scaled_p2)[0]
        result = 1 if pred1 == 1 and pred2 == 0 else (0 if pred1 == 0 and pred2 == 1 else -1)

    st.markdown("---")
    
    if result == 1:
        st.success(f"🏆 **Simulation Result:** **{team_a}** is predicted to win!")
    elif result == 0:
        st.info(f"🏆 **Simulation Result:** **{team_b}** is predicted to win!")
    else:
        st.warning(f"🤝 **Simulation Result:** The match is predicted to end in a **Draw**!")
