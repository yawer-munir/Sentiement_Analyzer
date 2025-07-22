import streamlit as st
from transformers import pipeline

# Load sentiment analysis pipeline from Hugging Face
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_model()

# Streamlit UI
st.title("🧠 Sentiment Analysis App")
st.write("Enter your text below and get the sentiment!")

user_input = st.text_area("Text input", placeholder="Type your sentence here...")

if st.button("Analyze"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = sentiment_pipeline(user_input)[0]
            st.success(f"**Label:** {result['label']} \n\n**Confidence:** {result['score']:.2f}")
    else:
        st.warning("Please enter some text.")
