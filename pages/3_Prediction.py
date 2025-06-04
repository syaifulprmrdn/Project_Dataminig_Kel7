import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Life Expectancy Prediction", page_icon="📈", layout="wide")
st.title("🚀 Life Expectancy Prediction Dashboard")
st.write("Masukkan data baru untuk memprediksi Life Expectancy.")

# Load dataset
df = pd.read_csv("Model/Life_Expectancy_Data.csv")

# Preprocessing data
df_clean = df.copy()
df_clean.drop(['Country'], axis=1, inplace=True)

le = LabelEncoder()
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

# Split data untuk training model
X = df_clean.drop(['Life expectancy '], axis=1)
y = df_clean['Life expectancy ']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training model di script
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Daftar fitur input
feature_list = X.columns.tolist()

# Input data user
st.subheader("📝 Masukkan Data untuk Prediksi")
input_data = {}
for feature in feature_list:
    if feature == 'Status':
        input_data[feature] = st.selectbox(f"{feature} (0=Developing, 1=Developed)", [0, 1])
    else:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Prediksi saat tombol ditekan
if st.button("Prediksi Life Expectancy"):
    input_df = pd.DataFrame([input_data])
    st.subheader("📁 Data yang Dimasukkan:")
    st.dataframe(input_df)

    prediksi = model.predict(input_df)
    st.success(f"🌟 Hasil Prediksi Life Expectancy: {prediksi[0]:.2f}")
