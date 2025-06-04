import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Life Expectanc Dashboard", layout="centered")
st.sidebar.header("Dashboard")

st.title("🎈 Selamat datang di Aplikasi Dashboard Data Life Expectancy")
st.write("Aplikasi ini memvisualisasikan data Life Expectancy.")

# Load Dataset
df = pd.read_csv("Model/Life Expectancy.csv")

# Tampilkan dataset dan kolom
st.subheader("📁 Dataset Life Expectancy")
st.write("Kolom:", df.columns.tolist())
st.dataframe(df)

# Visualisasi 1: Distribusi Life Expectancy
plt.figure(figsize=(8,5))
sns.histplot(data['Life expectancy '], bins=30, kde=True, color='skyblue')
plt.title('Distribusi Life Expectancy')
plt.xlabel('Life Expectancy')
plt.ylabel('Jumlah Negara')
plt.show()

# Visualisasi 2: Rata-rata Life Expectancy per Region
plt.figure(figsize=(12,6))
sns.barplot(data=data, x='Status', y='Life expectancy ')
plt.title('Rata-rata Life Expectancy Berdasarkan Status Negara')
plt.ylabel('Rata-rata Life Expectancy')
plt.xlabel('Status Negara')
plt.show()

# Visualisasi 3: Korelasi antar variabel numerik
plt.figure(figsize=(15,12))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Matriks Korelasi')
plt.show()

# Visualisasi 4: Tren Life Expectancy per Tahun
plt.figure(figsize=(10,6))
sns.lineplot(data=data, x='Year', y='Life expectancy ', ci=None)
plt.title('Tren Life Expectancy Global per Tahun')
plt.xlabel('Tahun')
plt.ylabel('Rata-rata Life Expectancy')
plt.show()
# Input interaktif
name = st.text_input("Siapa nama Anda?")
if name:
    st.success(f"Halo, {name}! 👋")

age = st.slider("Berapa umur Anda?", 0, 130, 25)
st.write("Umur Anda:", age, "tahun")
