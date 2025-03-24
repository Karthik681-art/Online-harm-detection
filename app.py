import streamlit as st 
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Label mapping from labeled_data.csv
label_map = {
    0: "Hate Speech",
    1: "Offensive",
    2: "Neutral"
}

st.title("AI-Powered Online Harm Detection")

text_input = st.text_area("Enter a message or comment")

if st.button("Detect"):
    if text_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        vec = vectorizer.transform([text_input])
        prediction = model.predict(vec)[0]
        label = label_map.get(prediction, f"Unknown label: {prediction}")
        st.success(f"Prediction: {label}")