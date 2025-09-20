import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and preprocessing tools
model = joblib.load("wine_quality_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

# -----------------------------------
# Title
# -----------------------------------
st.title("🍷 Wine Quality Prediction")

# Upload CSV option
st.subheader("📂 Bulk Prediction")
st.write("Upload a CSV file containing the 11 wine features (same format as the dataset).")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        expected_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                        'density', 'pH', 'sulphates', 'alcohol']

        if all(col in df.columns for col in expected_cols):
            X = df[expected_cols]

            # Preprocess
            X_imputed = imputer.transform(X)
            X_scaled = scaler.transform(X_imputed)

            # Predict probabilities and labels
            probas = model.predict_proba(X_scaled)[:, 1]
            preds = (probas >= 0.5).astype(int)

            # Save as percentages
            df['Probability_Good (%)'] = (probas * 100).round(2)
            df['Prediction'] = preds

            st.success("✅ Predictions completed!")
            view_option = st.radio(
                "Show rows:",
                ("Top 5", "Top 10", "All"),
                horizontal=True
            )

            if view_option == "Top 5":
                st.dataframe(df.head(5))
            elif view_option == "Top 10":
                st.dataframe(df.head(10))
            else:
                st.dataframe(df)

            # Download option
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="wine_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("⚠️ Uploaded file is missing one or more required columns.")
    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------------
# Manual Single Input
# -----------------------------------
st.subheader("🔹 Or Enter Wine Characteristics Manually")

# Default values for manual inputs only
default_manual_values = {
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.5,
    "citric_acid": 0.3,
    "residual_sugar": 2.0,
    "chlorides": 0.08,
    "free_sulfur": 15.0,
    "total_sulfur": 46.0,
    "density": 0.997,
    "pH": 3.3,
    "sulphates": 0.65,
    "alcohol": 10.0,
}

# Threshold default (kept separate)
if "threshold_percent" not in st.session_state:
    st.session_state["threshold_percent"] = 50

# Initialize session state for manual inputs
for key, val in default_manual_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Reset manual inputs button
if st.button("🔄 Reset Manual Inputs"):
    for key, val in default_manual_values.items():
        st.session_state[key] = val
    st.rerun()

# Layout with 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, st.session_state["fixed_acidity"], key="fixed_acidity")
    volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, st.session_state["volatile_acidity"], key="volatile_acidity")
    citric_acid = st.number_input("Citric Acid", 0.0, 2.0, st.session_state["citric_acid"], key="citric_acid")
    residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, st.session_state["residual_sugar"], key="residual_sugar")

with col2:
    chlorides = st.number_input("Chlorides", 0.0, 1.0, st.session_state["chlorides"], key="chlorides")
    free_sulfur = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, st.session_state["free_sulfur"], key="free_sulfur")
    total_sulfur = st.number_input("Total Sulfur Dioxide", 0.0, 200.0, st.session_state["total_sulfur"], key="total_sulfur")
    density = st.number_input("Density", 0.9, 1.5, st.session_state["density"], key="density")

with col3:
    pH = st.number_input("pH", 0.0, 14.0, st.session_state["pH"], key="pH")
    sulphates = st.number_input("Sulphates", 0.0, 2.0, st.session_state["sulphates"], key="sulphates")
    alcohol = st.number_input("Alcohol", 0.0, 20.0, st.session_state["alcohol"], key="alcohol")

# Reset threshold button (separate)
if st.button("🔄 Reset Threshold"):
    st.session_state["threshold_percent"] = 50
    st.rerun()

# Slider bound to session state
threshold_percent = st.slider(
    "Decision Threshold (%)",
    0,
    100,
    st.session_state["threshold_percent"],
    key="threshold_percent"
)

threshold = threshold_percent / 100

# Predict button
if st.button("Predict Manually"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol]])
    
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)

    # Get probability of good quality
    proba_good = model.predict_proba(features_scaled)[0][1]

    # Apply threshold
    prediction = 1 if proba_good >= threshold else 0

    # Show as percentage
    st.write(f"🔎 Probability of Good Quality: **{proba_good*100:.2f}%** (Threshold = {threshold*100:.0f}%)")

    if prediction == 1:
        st.success("✅ This wine is predicted to be **Good Quality**!")
    else:
        st.error("❌ This wine is predicted to be **Not Good Quality**.")
