import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Life Expectancy Prediction", page_icon="📈", layout="wide")
st.title("🚀 Life Expectancy Prediction Dashboard")
st.write("Pilih negara untuk prediksi Life Expectancy berdasarkan data rata-rata, atau masukkan data manual.")

# Load dataset
df = pd.read_csv("Model/Life_Expectancy_Data.csv")

# Preprocessing data
df_clean = df.copy()

# List negara unik
countries = df_clean['Country'].unique().tolist()

# Label encoding untuk fitur kategorikal selain Country
df_clean.drop(['Country'], axis=1, inplace=True)
le = LabelEncoder()
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

# Prepare train/test data
X = df_clean.drop(['Life expectancy '], axis=1)
y = df_clean['Life expectancy ']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Feature list
feature_list = X.columns.tolist()

# Pilih mode input
mode = st.radio("Pilih mode input data:", ("Pilih Negara untuk Prediksi Otomatis", "Input Data Manual"))

if mode == "Pilih Negara untuk Prediksi Otomatis":
    country = st.selectbox("Pilih Negara", countries)
    # Ambil rata-rata data negara yang dipilih
    country_data = df[df['Country'] == country]
    st.write(f"Data asli negara {country}:")
    st.dataframe(country_data)

    # Hitung rata-rata tiap fitur numerik dan kategorikal sudah di encode
    country_avg = country_data.mean(numeric_only=True)

    # Buat DataFrame untuk prediksi
    input_df = pd.DataFrame([country_avg[feature_list]])

    # Prediksi
    prediksi = model.predict(input_df)
    st.success(f"🌟 Prediksi Life Expectancy untuk negara **{country}** berdasarkan data rata-rata: {prediksi[0]:.2f}")

else:
    # Input manual (seperti sebelumnya)
    st.subheader("📝 Masukkan Data Manual untuk Prediksi")
    country_manual = st.text_input("Nama Negara")
    input_data = {}
    for feature in feature_list:
        if feature == 'Status':
            input_data[feature] = st.selectbox(f"{feature} (0=Developing, 1=Developed)", [0, 1])
        else:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Prediksi Life Expectancy"):
        input_df = pd.DataFrame([input_data])
        input_df['Country'] = country_manual  # hanya catatan saja
        st.subheader("📁 Data yang Dimasukkan:")
        st.dataframe(input_df[['Country'] + feature_list])

        prediksi = model.predict(input_df[feature_list])
        st.success(f"🌟 Hasil Prediksi Life Expectancy untuk **{country_manual}**: {prediksi[0]:.2f}")
