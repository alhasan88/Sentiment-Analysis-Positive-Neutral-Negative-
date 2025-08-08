import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load the saved model and vectorizer
model = joblib.load('model.pkl')
# Assumes vectorizer was saved
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to map POS tags to WordNet POS tags


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Preprocessing function


def preprocess_text(text):
    # Apply regex to remove symbols and keep words
    text = ' '.join(re.findall(r'\b[a-zA-Z]+\b', text.lower()))

    # Tokenize
    tokens = word_tokenize(text)

    # POS tagging and lemmatization
    pos_tags = nltk.pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(
        word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    # Remove stopwords
    filtered = [word for word in lemmatized if word not in stop_words]

    # Rejoin tokens
    processed_text = ' '.join(filtered)

    return processed_text


# Streamlit app configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Streamlit app


def main():
    st.title("Sentiment Analysis")
    st.write(
        "Enter text below to predict its sentiment (Negative, Neutral, or Positive).")

    # Text input
    user_input = st.text_area(
        "Your Text", placeholder="Enter your text here...", height=150)

    if st.button("Predict Sentiment"):
        if user_input:
            try:
                # Preprocess the text
                processed_text = preprocess_text(user_input)

                # Transform text using TF-IDF
                text_vector = tfidf_vectorizer.transform([processed_text])

                # Predict sentiment
                prediction = model.predict(text_vector)[0]

                # Map prediction to sentiment label
                sentiment_map = {-1.0: 'Negative',
                                 0.0: 'Neutral', 1.0: 'Positive'}
                result = sentiment_map.get(prediction, 'Unknown')

                # Display result
                st.markdown(
                    f'<div class="result">Predicted Sentiment: {result}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")


if __name__ == '__main__':
    # Save the TF-IDF vectorizer if not already saved
    if not os.path.exists('tfidf_vectorizer.pkl'):
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    main()
