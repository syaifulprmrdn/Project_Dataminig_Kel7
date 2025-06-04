import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Life Expectancy Model Dashboard", layout="wide")
st.title("🚀 Model Performance Dashboard: Life Expectancy")

# Load Dataset
df = pd.read_csv("Model/Life_Expectancy_Data.csv")
st.subheader("📁 Dataset Life Expectancy")
st.dataframe(df)

# Preprocessing
st.subheader("⚙️ Preprocessing Data")
df_clean = df.copy()
df_clean.drop(['Country'], axis=1, inplace=True)

le = LabelEncoder()
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
st.write("✅ Data setelah preprocessing:")
st.dataframe(df_clean.head())

# Train-test split
X = df_clean.drop(['Life expectancy '], axis=1)
y = df_clean['Life expectancy ']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
st.subheader("🧠 Training Model: Random Forest")
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi Model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write("### 🎯 Hasil Evaluasi Model:")
st.metric("R² Score", round(r2, 4))
st.metric("MSE", round(mse, 4))
st.metric("MAE", round(mae, 4))

# Visualisasi Prediksi vs Aktual
st.subheader("📊 Plot Prediksi vs Aktual")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color="green", ax=ax1)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_xlabel("Aktual Life Expectancy")
ax1.set_ylabel("Prediksi Life Expectancy")
ax1.set_title("Prediksi vs Aktual")
st.pyplot(fig1)

# Visualisasi Residual
st.subheader("📉 Plot Residual (Error)")
residuals = y_test - y_pred
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='orange', ax=ax2)
ax2.set_title("Distribusi Residual")
ax2.set_xlabel("Error (Aktual - Prediksi)")
st.pyplot(fig2)

# Feature Importance
st.subheader("🌟 Feature Importance")
importances = model.feature_importances_
feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
st.dataframe(feature_df)

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis', ax=ax3)
ax3.set_title("Feature Importance")
st.pyplot(fig3)
