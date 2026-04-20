import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.preprocessing import clean_text
from src.word2vec_demo import train_word2vec


# Load Word2Vec (cached)

@st.cache_resource
def load_model():
    return train_word2vec()

w2v_model = load_model()


# Rule-based spam detection

SPAM_KEYWORDS = [
    "free", "win", "winner", "cash", "prize",
    "claim", "urgent", "call", "offer", "reward"
]

def rule_based_predict(text):
    text = clean_text(text)
    score = 0

    matched_words = []

    for word in SPAM_KEYWORDS:
        if word in text:
            score += 1
            matched_words.append(word)

    if score >= 2:
        return "Spam", score, matched_words
    else:
        return "Ham", score, matched_words


# UI

st.set_page_config(page_title="NLP Spam Detector", page_icon="")

st.title(" NLP Spam Detection (Explainable)")
st.write("TF-IDF + Word2Vec based analysis")


# Input

user_input = st.text_area("Enter your message:")


# Prediction

if st.button("Analyze"):

    if not user_input.strip():
        st.warning("Please enter a message")
    else:
        label, score, matched = rule_based_predict(user_input)

        if label == "Spam":
            st.error("Spam Detected")
        else:
            st.success("Ham Message")

        st.write(f"**Spam Score:** {score}")

        st.write("### 🔍 Detected Keywords:")
        st.write(matched if matched else "No strong spam keywords found")


# Word2Vec Demo

st.markdown("---")
st.subheader(" Word Similarity (Word2Vec)")

word = st.text_input("Enter a word (e.g., free, win, call)")

if word:
    if word in w2v_model.wv:
        similar_words = w2v_model.wv.most_similar(word, topn=5)
        st.write(similar_words)
    else:
        st.warning("Word not in vocabulary")


# Examples

with st.expander(" Try Examples"):
    st.write("Spam: Congratulations! You've won a $1000 gift card!")
    st.write("Ham: Hey, are we meeting at 6?")