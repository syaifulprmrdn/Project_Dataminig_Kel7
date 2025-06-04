import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Prediction", layout="wide")
st.title("🔮 Life Expectancy Prediction")

# Load data

# Preprocess
target_col = "Diagnosis"
X = df.drop(columns=[target_col])
y = df[target_col]
if y.dtypes == 'object':
    y = pd.factorize(y)[0]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Input user
st.subheader("Masukkan data pasien:")
input_data = {}
for col in X.columns:
    val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    input_data[col] = val

if st.button("Prediksi Diagnosis"):
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    st.success(f"Prediksi diagnosis: {pred}")
