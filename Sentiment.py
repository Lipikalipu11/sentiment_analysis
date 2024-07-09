 import streamlit as st
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the trained TF-IDF vectorizer and SVM model
vector = pickle.load(open('./model/vectorized.pkl', 'rb'))
model = pickle.load(open('./model/ensemble_model.pkl', 'rb'))

def preprocessor(review):
    # Same preprocessor function as before
    review = HTMLTAGS.sub(r'', review)
    review = review.translate(table)
    review = review.translate(remove_digits)
    review = review.lower()
    review = MULTIPLE_WHITESPACE.sub(" ", review).strip()
    review = [word for word in review.split() if word not in final_stopwords]
    review = ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in review])
    return review

def get_sentiment(review):
    processed_review = preprocessor(review)
    x = vectorizer.transform([processed_review])
    y = ensemble_model.predict(x)
    return y[0]

# Streamlit app title and description
st.title('Amazon Product Review Sentiment Analysis')
st.write('Enter a product review to predict its sentiment (Positive, Neutral, or Negative).')

# Text input for the user to enter a review
user_input = st.text_area('Enter a product review:')

# Predict sentiment when button is pressed
if st.button('Predict Sentiment'):
    if user_input.strip() != '':
        sentiment = predict_sentiment(user_input)
        st.write(f'The sentiment is: *{sentiment}*')
    else:
        st.write("Please enter a review to predict.")

# Optional: Add information about the model
st.sidebar.title("About")
st.sidebar.write("""
This app uses a Support Vector Machine (SVM) model trained on Amazon product reviews to classify the sentiment of new reviews as positive, neutral, or negative. The text input is processed using TF-IDF vectorization before being fed into the model.
""")
