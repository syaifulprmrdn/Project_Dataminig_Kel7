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

# Tampilkan tabel
st.subheader("Tabel Rata-rata Life Expectancy per Negara")
st.dataframe(avg_life_expectancy.sort_values(by='Life expectancy ', ascending=False).reset_index(drop=True))

# Visualisasi Bar Plot
st.subheader("Visualisasi Rata-rata Life Expectancy per Negara")
plt.figure(figsize=(15,8))
sns.barplot(data=avg_life_expectancy.sort_values(by='Life expectancy ', ascending=False),
            x='Life expectancy ', y='Country', palette='viridis')
plt.xlabel("Rata-rata Life Expectancy")
plt.ylabel("Negara")
plt.title("Rata-rata Life Expectancy per Negara")
st.pyplot(plt)
