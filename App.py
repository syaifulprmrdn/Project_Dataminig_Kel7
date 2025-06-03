import streamlit as st
import pandas as pd
import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Iris Dashboard App", layout="centered")
st.sidebar.header("Dashboard")

st.title("🎈 Selamat datang di Aplikasi Streamlit Sederhana")
st.write("Aplikasi ini dibuat untuk demonstrasi projek akhir Data Mining.")

#Load Dataset
df = pd.read_csv("model/iris.csv")

# Tampilkan dataframe
st.subheader("📁 Dataset Iris")
st.dataframe(df)

st.write(df.columns.tolist())
#df[target] = data target
#df['label'] = df['variety'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})
class_counts = df['variety'].value_counts()

#Distribusi Kelas
st.subheader("Distribusi Jumlah Data Berdasarkan Kelas")
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette=["red", "green", "yellow"], ax=ax)
ax.set_ylabel("Jumlah Data")
ax.set_xlabel("Varietas")
ax.set_title("Distribusi Kelas Iris")
st.pyplot(fig)

#Korelasi Fitur
st.subheader("Korelasi antar Fitur dalam Dataset")

# Input interaktif
name = st.text_input("Siapa nama Anda?")
if name:
    st.success(f"Halo, {name}! 👋")

age = st.slider("How old are you?", 0, 130, 25)
st.write("I'm ", age, "years old")
