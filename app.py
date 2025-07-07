import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("movie_reviews.csv")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

# Apply cleaning
df["cleaned_review"] = df["review"].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["cleaned_review"])

# Label encoding
le = LabelEncoder()
y = le.fit_transform(df["sentiment"])

# Model
model = LogisticRegression()
model.fit(X, y)

# Streamlit App
st.title("🎬 Movie Review Sentiment Analyzer")

user_input = st.text_area("Enter your movie review below:")

if st.button("Analyze Sentiment"):
    cleaned = clean_text(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    sentiment = le.inverse_transform([prediction])[0]

    if sentiment == "positive":
        st.success(" Sentiment: Positive")
    else:
        st.error("Sentiment: Negative")
