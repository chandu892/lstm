import streamlit as st
import numpy as np
import pickle

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="📖"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"

@st.cache_resource
def load_artifacts():

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_artifacts()

max_seq_len = 20   # replace with training value

st.title("📖 Next Word Prediction using LSTM")

seed_text = st.text_input(
    "Enter a sentence",
    "to be or not to"
)

num_words = st.slider(
    "Number of words to generate",
    1,
    10,
    3
)

def predict_next_words(seed_text, num_words):

    output = seed_text

    for _ in range(num_words):

        token_list = tokenizer.texts_to_sequences([output])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_seq_len - 1,
            padding="pre"
        )

        predicted = model.predict(
            token_list,
            verbose=0
        )

        predicted_index = np.argmax(predicted)

        next_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                next_word = word
                break

        output += " " + next_word

    return output

if st.button("Generate"):

    result = predict_next_words(
        seed_text,
        num_words
    )

    st.success(result)