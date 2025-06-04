import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediction", page_icon="📈")
st.header("Prediction")
st.write("Make a prediction using a new data")

@st.cache_resource
def load_model(path):
    model = joblib.load(path)
    return model

model = load_model('model/decision_tree_model.joblib')

st.write("Masukkan data yang akan diprediksi:")

# Input fitur
seplen = st.number_input("Sepal Length", min_value=0.0, max_value=8.0, value=2.0)
sepwid = st.number_input("Sepal Width", min_value=0.0, max_value=8.0, value=2.0)
petlen = st.number_input("Petal Length", min_value=0.0, max_value=8.0, value=2.0)
petwid = st.number_input("Petal Width", min_value=0.0, max_value=8.0, value=2.0)

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    input_data = pd.DataFrame([[seplen, sepwid, petlen, petwid]],
                              columns=["sepal.length", "sepal.width", "petal.length", "petal.width"])
    st.dataframe(input_data)
    hasil = model.predict(input_data)
    st.success(f"Hasil Prediksi: {hasil[0]}")
