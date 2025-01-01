import streamlit as st
import requests

st.title("Spam Message Classification")

message = st.text_area("Enter your message:")
model = st.selectbox("Choose a model:", ["Random Forest", "Support Vector Machine"])

if st.button("Predict"):
    model_filename = "model_random_Forest_Classifier.pkl" if model == "Random Forest" else "model_svm.pkl"
    response = requests.post(
        "http://127.0.0.1:8000/predict/",
        data={"message": message, "model": model_filename}
    )
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
    else:
        st.error(f"Error: {response.json().get('error')}")
