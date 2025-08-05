# app.py

import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the dataset for analytics
df = pd.read_csv("IMDB Dataset.csv")
df.dropna(inplace=True)

# Convert sentiments to lowercase (just in case)
df['sentiment'] = df['sentiment'].str.lower()

# Count positive and negative reviews
sentiment_counts = df['sentiment'].value_counts()

# Streamlit page config
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered", initial_sidebar_state="collapsed")

# Custom dark input CSS
st.markdown("""
    <style>
        .stTextInput > div > div > input {
            background-color: #2c2f33;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("## 🎯 IMDB Sentiment Analysis")
st.write("Enter a review to predict its sentiment based on your dataset.")
review = st.text_input("Type your review here:")
if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"**Sentiment:** {sentiment}")


# 🔍 Analytics Chart
st.markdown("### 🔎 Review Sentiment Analytics")

fig, ax = plt.subplots()
ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red'])
ax.set_ylabel("Number of Reviews")
ax.set_xlabel("Sentiment")
ax.set_title("Sentiment Distribution")
st.pyplot(fig)

