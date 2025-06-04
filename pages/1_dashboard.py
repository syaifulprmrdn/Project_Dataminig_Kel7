import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Rata-rata Life Expectancy per Negara", layout="wide")
st.title("📊 Rata-rata Life Expectancy per Negara")

# Load dataset
df = pd.read_csv("Model/Life_Expectancy_Data.csv")

# Hitung rata-rata Life Expectancy per negara
avg_life_expectancy = df.groupby('Country')['Life expectancy '].mean().reset_index()

# Sort descending dan ambil top 20 negara
top_n = 20
top_countries = avg_life_expectancy.sort_values(by='Life expectancy ', ascending=False).head(top_n)

# Tampilkan tabel top 20
st.subheader(f"Tabel Top {top_n} Negara dengan Rata-rata Life Expectancy Tertinggi")
st.dataframe(top_countries.reset_index(drop=True))

# Visualisasi Bar Plot Top 20
st.subheader(f"Visualisasi Top {top_n} Negara dengan Rata-rata Life Expectancy Tertinggi")
plt.figure(figsize=(12,8))
sns.barplot(
    data=top_countries,
    y='Country',
    x='Life expectancy ',
    palette='mako',
    orient='h'
)
plt.xlabel("Rata-rata Life Expectancy")
plt.ylabel("Negara")
plt.title(f"Top {top_n} Negara dengan Rata-rata Life Expectancy Tertinggi")
plt.xlim(0, top_countries['Life expectancy '].max() + 5)
plt.grid(axis='x', linestyle='--', alpha=0.7)
st.pyplot(plt)
