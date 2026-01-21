# app.py
import streamlit as st
import pickle

# ----------------------------
# Title
# ----------------------------
st.set_page_config(page_title="Multi-Disease Prediction App", page_icon="🩺", layout="centered")
st.title("🩺 Multi-Disease Symptom-Based Prediction")
st.write("Enter your symptoms and the app will predict the most likely disease.")

# ----------------------------
# Load Model and Vectorizer
# ----------------------------
try:
    with open("disease_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

except FileNotFoundError:
    st.error("Model files not found. Make sure disease_model.pkl and tfidf_vectorizer.pkl are in the same folder as app.py")
    st.stop()

# ----------------------------
# User Input
# ----------------------------
user_input = st.text_area("Enter symptoms (e.g., fever, headache, cough):")

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Disease"):
    if user_input.strip() == "":
        st.warning("Please enter some symptoms to predict the disease!")
    else:
        # Convert input text to vector
        input_vect = vectorizer.transform([user_input])
        # Predict disease
        prediction = model.predict(input_vect)
        st.success(f"Predicted Disease: **{prediction[0]}**")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("**Note:** This is a demo ML app. For medical advice, always consult a doctor.")
