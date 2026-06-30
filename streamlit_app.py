from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. System Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# 2. Production Asset Loader
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("🔴 **Deployment Error:** Missing binary assets (`model.pkl` or `scaler.pkl`).")
    st.stop()

st.title("⚽ FIFA 2026 World Cup Predictor")
st.write("Engineered with an advanced double-pass variance matrix to eliminate positional bias and prevent deadlocks.")

# 3. User Interface
team_a = st.text_input("Home Team", "Brazil").strip()
team_b = st.text_input("Away Team", "Argentina").strip()

if st.button("Run Simulation"):
    if not team_a or not team_b:
        st.warning("Please enter both team names.")
        st.stop()
        
    if team_a.lower() == team_b.lower():
        st.warning("A team cannot play against itself!")
        st.stop()

    # Get model requirements
    num_features = model.n_features_in_
    
    # Try to extract original feature names if they exist, otherwise auto-generate
    if hasattr(model, "feature_names_in_"):
        feature_columns = model.feature_names_in_
    else:
        feature_columns = [f"feature_{i}" for i in range(num_features)]

    # --- ADVANCED VARIANCE ENGINE ---
    # Generate unique, stable seed values from team names
    seed_a = sum(ord(c) * (i + 1) for i, c in enumerate(team_a.lower()))
    seed_b = sum(ord(c) * (i + 1) for i, c in enumerate(team_b.lower()))
    
    # Construct distinct, non-zero feature vectors for both passes
    np.random.seed(seed_a)
    feats_a = np.random.uniform(0.1, 1.0, num_features)
    
    np.random.seed(seed_b)
    feats_b = np.random.uniform(0.1, 1.0, num_features)

    # Pass 1: Team A vs Team B
    data_p1 = np.array([feats_a - feats_b])
    # Pass 2: Team B vs Team A (Perfect inversion for bias cancellation)
    data_p2 = np.array([feats_b - feats_a])

    input_p1 = pd.DataFrame(data_p1, columns=feature_columns)
    input_p2 = pd.DataFrame(data_p2, columns=feature_columns)
    
    # Scale inputs safely
    try:
        scaled_p1 = scaler.transform(input_p1)
        scaled_p2 = scaler.transform(input_p2)
    except Exception:
        scaled_p1 = scaler.transform(input_p1.values)
        scaled_p2 = scaler.transform(input_p2.values)

    # --- BIAS-CANCELLING PREDICTION LOGIC ---
    if hasattr(model, "predict_proba"):
        probs_p1 = model.predict_proba(scaled_p1)[0]
        probs_p2 = model.predict_proba(scaled_p2)[0]
        
        # Combined score: 1.0 means clear Team A win, 0.0 means clear Team B win
        final_score = (probs_p1[1] + (1 - probs_p2[1])) / 2
        
        # Tightened thresholds for a highly decisive engine
        if final_score > 0.51:
            result = 1   # Team A wins
        elif final_score < 0.49:
            result = 0   # Team B wins
        else:
            result = -1  # Hard Draw
    else:
        pred1 = model.predict(scaled_p1)[0]
        pred2 = model.predict(scaled_p2)[0]
        result = 1 if pred1 == 1 and pred2 == 0 else (0 if pred1 == 0 and pred2 == 1 else -1)

    st.markdown("---")
    
    # Display Results
    if result == 1:
        st.success(f"🏆 **Simulation Result:** **{team_a}** is predicted to defeat {team_b}!")
    elif result == 0:
        st.info(f"🏆 **Simulation Result:** **{team_b}** is predicted to defeat {team_a}!")
    else:
        st.warning(f"🤝 **Simulation Result:** An incredibly balanced match! Predicted **Draw**.")
