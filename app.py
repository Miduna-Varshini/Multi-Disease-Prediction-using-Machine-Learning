# app.py
import streamlit as st
import pickle
import pandas as pd

# Title
st.title("Multi-Disease Prediction App")
st.write("Enter your symptoms and the app will predict the most likely disease.")

# Load the saved model and vectorizer
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# User input
user_input = st.text_area("Enter symptoms (e.g., fever, headache, cough):")

if st.button("Predict Disease"):
    if user_input.strip() == "":
        st.warning("Please enter some symptoms!")
    else:
        # Transform input using TF-IDF
        input_vect = vectorizer.transform([user_input])
        # Predict
        prediction = model.predict(input_vect)
        st.success(f"Predicted Disease: {prediction[0]}")
