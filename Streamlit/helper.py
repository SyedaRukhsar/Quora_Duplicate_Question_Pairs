import re
import tensorflow as tf
import pickle
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import keras

@keras.saving.register_keras_serializable()
class AbsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)



@st.cache_data
def load_tokenizer(path="Streamlit/tokenizer.pkl"):
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Model load function
def load_model(path="siamese_bilstm_final.keras"):
    model = keras.models.load_model(path, custom_objects={"AbsLayer": AbsLayer})
    return model



def preprocess(q):
    q = str(q).lower().strip()

    # Replace special characters
    q = q.replace('%', 'percent').replace('$', 'dollar').replace('@', 'at').replace('₹', 'rupee').replace('€', 'euro')

    # Remove math token
    q = q.replace('[math]', '')

    # Numbers to k/m/b
    q = q.replace(',000,000,000', 'b').replace(',000,000', 'm').replace(',000', 'k')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Contractions
    contractions = {
        "ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "cause": "because",
        "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am", "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us",
        "ma'am": "madam", "mightn't": "might not", "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
        "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are",
        "they've": "they have",
        "wasn't": "was not", "we'd": "we would", "we're": "we are", "we've": "we have", "weren't": "were not",
        "what'll": "what will",
        "what're": "what are", "what's": "what is", "what've": "what have", "where's": "where is", "who's": "who is",
        "won't": "will not", "would've": "would have", "wouldn't": "would not", "you'd": "you would",
        "you'll": "you will", "you're": "you are"
    }
    # Cleaner function
    REPLACE_BY_SPACE_RE = re.compile(r'[\t\n\r]+')
    BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z ]')

    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
    q = " ".join(q_decontracted)

    # Remove HTML tags
    q = BeautifulSoup(q, "html.parser").get_text()

    # Remove punctuation
    q = re.sub(r'\W', ' ', q).strip()

    # replace newlines and tabs with space
    q = REPLACE_BY_SPACE_RE.sub(' ', q)
    # remove unwanted characters (keep a-z and numbers)
    q = BAD_SYMBOLS_RE.sub(' ', q)
    # collapse multiple spaces
    q = re.sub(' +', ' ', q).strip()

    return q



# calculate max length
max_len = 30

# Initialize global objects
tokenizer = load_tokenizer("Streamlit/tokenizer.pkl")
model = load_model("siamese_bilstm_final.keras")



def preprocess_single(q):
    q = preprocess(q)
    seq = tokenizer.texts_to_sequences([q])
    pad = pad_sequences(seq, maxlen=30, padding='post')
    return pad


def predict_pair(q1, q2, thresh=0.5):
    s1 = preprocess_single(q1)
    s2 = preprocess_single(q2)
    p = model.predict([s1, s2])[0, 0]
    return {'probability': float(p), 'is_duplicate': int(p >= thresh)}


