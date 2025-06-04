import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from geopy.geocoders import Nominatim

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

# Daftar fitur input (selain Country yang hanya disimpan)
feature_list = X.columns.tolist()

# Input data user
st.subheader("📝 Masukkan Data untuk Prediksi")
country = st.text_input("Nama Negara")
input_data = {}
for feature in feature_list:
    if feature == 'Status':
        input_data[feature] = st.selectbox(f"{feature} (0=Developing, 1=Developed)", [0, 1])
    else:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Prediksi saat tombol ditekan
if st.button("Prediksi Life Expectancy"):
    input_df = pd.DataFrame([input_data])
    input_df['Country'] = country  # Tambahkan kolom nama negara
    st.subheader("📁 Data yang Dimasukkan:")
    st.dataframe(input_df[['Country'] + feature_list])

    # Prediksi
    prediksi = model.predict(input_df[feature_list])
    st.success(f"🌟 Hasil Prediksi Life Expectancy untuk **{country}**: {prediksi[0]:.2f}")

    # Cari koordinat negara
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(country)
    if location:
        lat, lon = location.latitude, location.longitude
        st.subheader("🗺️ Lokasi Negara:")
        st.write(f"Latitude: {lat}, Longitude: {lon}")

        # Buat data peta
        map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        st.pydeck_chart(
            {
                "layers": [
                    {
                        "type": "ScatterplotLayer",
                        "data": map_data,
                        "get_position": "[lon, lat]",
                        "get_radius": 1000000,
                        "get_fill_color": [255, 0, 0, 160],
                        "pickable": True,
                    }
                ],
                "initialViewState": {
                    "latitude": lat,
                    "longitude": lon,
                    "zoom": 2,
                    "pitch": 0,
                },
                "mapStyle": "mapbox://styles/mapbox/light-v9",
            }
        )
    else:
        st.warning("❌ Lokasi negara tidak ditemukan.")
