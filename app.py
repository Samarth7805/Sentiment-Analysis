

import streamlit as st
from model import predict_sentiment

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭")


st.title("🎭 Sentiment Analyzer")
st.write("Analyze the sentiment of any text - Positive, Negative, or Neutral")


user_text = st.text_area("Enter your text below:", height=120)

if st.button("Analyze Sentiment"):
    if user_text.strip():

        prediction, confidence, reason = predict_sentiment(user_text)

        
        if prediction == "positive":
            st.success(f"😊 Sentiment: **POSITIVE** (Confidence: {confidence:.2f})")
            st.write(f"**Reason:** {reason}")
        elif prediction == "negative":
            st.error(f"😞 Sentiment: **NEGATIVE** (Confidence: {confidence:.2f})")
            st.write(f"**Reason:** {reason}")
        else:
            st.info(f"😐 Sentiment: **NEUTRAL** (Confidence: {confidence:.2f})")
            st.write(f"**Reason:** {reason}")
    else:
        st.warning("Please enter some text first.")


