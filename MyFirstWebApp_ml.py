import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from pickle import dump ,load

classifier_loc = "logit_model.pkl"
encoder_loc = "countvectorizer.pkl"
image_loc = "message-1.jpg"


def preprocess(msg):
    # Removing special characters and digits
    letters_only = re.sub("[^a-zA-Z]", " ",msg)

    # change sentence to lower case
    letters_only = letters_only.lower()

    # tokenize into words
    words = letters_only.split()

    # remove stop words
    words = [w for w in words if not w in stopwords.words("english")]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    clean_sentence = " ".join(words)

    return clean_sentence


def predict(msg):

    # Loading pretrained CountVectorizer from pickle file
    vectorizer = load(open('countvectorizer.pkl', 'rb'))

    # Loading pretrained logistic classifier from pickle file
    classifier = load(open('logit_model.pkl', 'rb'))

    # Preprocessing the tweet
    clean_msg = preprocess(msg)

    # Converting text to numerical vector
    clean_msg_encoded = vectorizer.transform([clean_msg])

    # Converting sparse matrix to dense matrix
    msg_input = clean_msg_encoded.toarray()

    # Prediction
    prediction = classifier.predict(msg_input)

    return prediction


def main():

    st.image("message-1.jpg", use_column_width = True)

    msg = st.text_input('Enter your message')

    prediction = predict(msg)

    if(msg):
        st.subheader("Prediction:")
        if(prediction == 0):
            st.write("spam message :cry:")
        else:
            st.write("ham message :sunglasses:")



if(__name__ == '__main__'):
    main()





