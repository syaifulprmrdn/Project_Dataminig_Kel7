import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="Life Expectancy Dashboard", layout="centered")
st.sidebar.header("Dashboard")

st.title("🎈 Selamat datang di Aplikasi Dashboard Data Life Expectancy")
st.write("Aplikasi ini memvisualisasikan data Life Expectancy.")

# Load Dataset
df = pd.read_csv("Model/Life_Expectancy_Data.csv")

# Tampilkan dataset dan kolom
st.subheader("📁 Dataset Life Expectancy")
st.write("Kolom:", df.columns.tolist())
st.dataframe(df)

# Visualisasi 1: Distribusi Life Expectancy
st.subheader("Distribusi Life Expectancy")
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df['Life expectancy '], bins=30, kde=True, color='skyblue', ax=ax)
ax.set_title('Distribusi Life Expectancy')
ax.set_xlabel('Life Expectancy')
ax.set_ylabel('Jumlah Negara')
st.pyplot(fig)

# Visualisasi 2: Rata-rata Life Expectancy per Status Negara
st.subheader("Rata-rata Life Expectancy Berdasarkan Status Negara")
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(data=df, x='Status', y='Life expectancy ', ax=ax)
ax.set_title('Rata-rata Life Expectancy Berdasarkan Status Negara')
ax.set_ylabel('Rata-rata Life Expectancy')
ax.set_xlabel('Status Negara')
st.pyplot(fig)

# Visualisasi 3: Korelasi antar variabel numerik
st.subheader("Matriks Korelasi antar Variabel Numerik")
fig, ax = plt.subplots(figsize=(15,12))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Matriks Korelasi')
st.pyplot(fig)

# Visualisasi 4: Tren Life Expectancy per Tahun
st.subheader("Tren Life Expectancy Global per Tahun")
fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(data=df, x='Year', y='Life expectancy ', ci=None, ax=ax)
ax.set_title('Tren Life Expectancy Global per Tahun')
ax.set_xlabel('Tahun')
ax.set_ylabel('Rata-rata Life Expectancy')
st.pyplot(fig)

# Input interaktif
st.subheader("👤 Informasi Pengguna")
name = st.text_input("Siapa nama Anda?")
if name:
    st.success(f"Halo, {name}! 👋")

age = st.slider("Berapa umur Anda?", 0, 130, 25)
st.write("Umur Anda:", age, "tahun")
