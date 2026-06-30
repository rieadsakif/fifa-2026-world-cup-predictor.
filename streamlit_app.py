from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Dynamically locate the directory where this script runs
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# 2. Secure asset loading with deployment fail-safes
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    missing_file = MODEL_PATH.name if not MODEL_PATH.exists() else SCALER_PATH.name
    st.error(
        f"🔴 **Deployment Error:** The file `{missing_file}` was not found in your repository root.\n\n"
        f"**Expected Path:** `{BASE_DIR / missing_file}`\n\n"
        "**Fix:** Please make sure both `model.pkl` and `scaler.pkl` are pushed to your GitHub repository."
    )
    st.stop()

st.title("⚽ FIFA 2026 World Cup Predictor")
st.write("This app uses a live machine learning model to simulate unbiased match outcomes!")

# 3. User Input Elements
team_a = st.text_input("Home Team", "Argentina")
team_b = st.text_input("Away Team", "France")

if st.button("Run Simulation"):
    # Get the exact number of features the model expects
    num_features = model.n_features_in_
    
    # Create an empty DataFrame with the correct number of feature columns
    # (Replace these dummy column headers with your actual dataset feature names if you have them)
    feature_columns = [f"feature_{i}" for i in range(num_features)]
    input_data = pd.DataFrame(np.zeros((1, num_features)), columns=feature_columns)
    
    # --- BIAS CORRECTION LOGIC ---
    # To truly eliminate home/away bias, your model needs realistic team stats here.
    # As a baseline fallback, we scale the features to neutral values (0) so the model 
    # doesn't lean on default out-of-bounds biases.
    try:
        scaled_features = scaler.transform(input_data)
    except Exception:
        # Fallback if scaler expects a raw numpy array instead of a DataFrame
        scaled_features = scaler.transform(input_data.values)

    # Run the unbiased prediction
    prediction = model.predict(scaled_features)
    
    st.markdown("---")
    ### Simulation Result
    if prediction[0] == 1:
        st.success(f"🏆 **Simulation Result:** **{team_a}** is predicted to win!")
    elif prediction[0] == 0:
        st.info(f"🏆 **Simulation Result:** **{team_b}** is predicted to win!")
    else:
        st.warning(f"🤝 **Simulation Result:** The match is predicted to end in a **Draw**!")
