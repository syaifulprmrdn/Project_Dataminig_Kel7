import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Life Expectancy Prediction", page_icon="📈", layout="wide")
st.title("🚀 Life Expectancy Prediction Dashboard")
st.write("Pilih negara untuk prediksi Life Expectancy berdasarkan data rata-rata, atau masukkan data manual.")

# Load dataset asli
df = pd.read_csv("Model/Life_Expectancy_Data.csv")

# Copy dataframe untuk preprocessing
df_clean = df.copy()

# List negara unik (dari dataset asli)
countries = df['Country'].unique().tolist()

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

# Fungsi untuk preprocessing data input negara
def preprocess_country_data(country_name):
    # Filter data asli sesuai negara
    country_df = df[df['Country'] == country_name]
    if country_df.empty:
        return None
    # Hitung rata-rata tiap kolom numerik dan kategorikal yang perlu di-encode
    avg_data = country_df.mean(numeric_only=True).to_dict()

    # Untuk fitur kategorikal (object), lakukan encoding manual:
    # Karena kita hanya punya 'Status' sebagai fitur kategorikal lain setelah drop Country
    # Kita bisa cari mode Status di negara tsb dan encode secara manual
    status_mode = country_df['Status'].mode()
    if not status_mode.empty:
        avg_data['Status'] = 1 if status_mode[0].lower() == 'developed' else 0
    else:
        avg_data['Status'] = 0  # default

    # Pastikan semua fitur ada di avg_data, jika tidak beri nilai 0 atau median
    input_features = {}
    for feat in feature_list:
        input_features[feat] = avg_data.get(feat, 0)

    return pd.DataFrame([input_features])

# Pilih mode input
mode = st.radio("Pilih mode input data:", ("Pilih Negara untuk Prediksi Otomatis", "Input Data Manual"))

if mode == "Pilih Negara untuk Prediksi Otomatis":
    country = st.selectbox("Pilih Negara", countries)

    if country:
        st.write(f"Data asli negara {country}:")
        st.dataframe(df[df['Country'] == country])

        input_df = preprocess_country_data(country)
        if input_df is not None:
            prediksi = model.predict(input_df)
            st.success(f"🌟 Prediksi Life Expectancy untuk negara **{country}** berdasarkan data rata-rata: {prediksi[0]:.2f}")
        else:
            st.error("Data negara tidak ditemukan atau tidak cukup lengkap untuk prediksi.")

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
        st.subheader("📁 Data yang Dimasukkan:")
        st.dataframe(input_df)

        prediksi = model.predict(input_df)
        st.success(f"🌟 Hasil Prediksi Life Expectancy untuk **{country_manual}**: {prediksi[0]:.2f}")
