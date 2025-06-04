import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("📈 Model Performance")

# Load data


# Preprocess
target_col = "Diagnosis"
X = df.drop(columns=[target_col])
y = df[target_col]
if y.dtypes == 'object':
    y = pd.factorize(y)[0]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

st.subheader("📄 Classification Report")
st.dataframe(df_report)

# Confusion matrix
st.subheader("📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)
