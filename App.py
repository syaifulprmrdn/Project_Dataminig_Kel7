import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="China Cancer Patients Dashboard", layout="centered")
st.sidebar.header("Dashboard")

st.title("🎈 Selamat datang di Aplikasi Dashboard Cancer Patients")
st.write("Aplikasi ini memvisualisasikan data synthetic China cancer patients.")

# Load Dataset
df = pd.read_csv("Model/china_cancer_patients_synthetic.csv")

# Tampilkan dataset dan kolom
st.subheader("📁 Dataset Cancer Patients")
st.write("Kolom:", df.columns.tolist())
st.dataframe(df)

# Distribusi kelas (jika ada kolom target)
if 'Diagnosis' in df.columns:
    st.subheader("Distribusi Jumlah Data Berdasarkan Diagnosis")
    diagnosis_counts = df['Diagnosis'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values, palette="Set2", ax=ax)
    ax.set_ylabel("Jumlah Data")
    ax.set_xlabel("Diagnosis")
    ax.set_title("Distribusi Kelas Diagnosis")
    st.pyplot(fig)
else:
    st.write("Kolom target 'Diagnosis' tidak ditemukan.")

# Korelasi fitur numerik
st.subheader("Korelasi antar Fitur Numerik")
numerical_df = df.select_dtypes(include=['float64', 'int64'])
if not numerical_df.empty:
    corr = numerical_df.corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
else:
    st.write("Tidak ada kolom numerik untuk dihitung korelasinya.")

# Input interaktif
name = st.text_input("Siapa nama Anda?")
if name:
    st.success(f"Halo, {name}! 👋")

age = st.slider("Berapa umur Anda?", 0, 130, 25)
st.write("Umur Anda:", age, "tahun")
