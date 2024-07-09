import pickle
import re
import string
# Install NLTK
!pip install nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load necessary resources (vectorizer and model)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('ensemble_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

# Define preprocessor functions and resources
HTMLTAGS = re.compile(r'<.*?>')
table = str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
remove_digits = str.maketrans('', '', '0123456789')
MULTIPLE_WHITESPACE = re.compile(r"\s+")
lemmatizer = WordNetLemmatizer()
final_stopwords = set(stopwords.words('english'))

def preprocessor(review):
    review = HTMLTAGS.sub(r'', review)
    review = review.translate(table)
    review = review.translate(remove_digits)
    review = review.lower()
    review = MULTIPLE_WHITESPACE.sub(" ", review).strip()
    review = [word for word in review.split() if word not in final_stopwords]
    review = ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in review])
    return review

def get_wordnet_pos(word):
    from nltk.corpus import wordnet
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def get_sentiment(review):
    processed_review = preprocessor(review)
    x = vectorizer.transform([processed_review])
    y = ensemble_model.predict(x)
    return y[0]

if __name__ == "__main__":
    # Example usage
    review = "I excellent and love this laptop"
    sentiment = get_sentiment(review)
    print(f"This is a {sentiment} sentiment!")
