import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Prediction", page_icon="📈")
st.header("Predictions")

#Load Dataset
df = pd.read_csv("model/iris.csv")
dataset = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')

#Tampilkan Dataframe
st.subheader("📁 Iris Dataset")
st.dataframe(dataset)

#Test Size
testing = st.slider("Data Testing", min_value=10, max_value=90, value=20)
st.write(f"Nilai yang dipilih: {testing}")
t_size = testing/100

#Select features and target
X = dataset.drop('variety', axis=1)
y = dataset['variety']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=42)

#Load Model
@st.cache_resource
def load_model(path):
    model = joblib.load(path)
    return model

model1 = load_model('model/decision_tree_model.joblib')

# Prediksi saat tombol ditekan
if st.button("Hasil"):
    y_pred1 = model1.predict(X_test)
    #Evaluate the models
    accuracy1 = accuracy_score(y_test, y_pred1)
    metric1 = classification_report(y_test, y_pred1, output_dict=True)
    st.success(f"Model {type(model1).__name__}")

    a, b = st.columns(2)
    c, d = st.columns(2)

    acc1 = metric1["accuracy"]*100
    prec1 = metric1["macro avg"]["precision"]*100
    a.metric("Accuracy", f"{acc1}%", delta=None, border=True)
    b.metric("Precision", f"{prec1}%", delta=None, border=True)

    rec1 = metric1["macro avg"]["recall"]*100
    fsc1 = metric1["macro avg"]["f1-score"]*100
    c.metric("Recall", f"{rec1}%", delta=None, border=True)
    d.metric("F1-Score", f"{fsc1}%", delta=None, border=True)
