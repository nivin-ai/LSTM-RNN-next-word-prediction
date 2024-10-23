import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import streamlit as st


# load the model
model = load_model('next_word_lstm.h5')

# load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] #to ensure that the sequence length matches the max sequence length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict(token_list, verbose=1)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None


## Streamlit app
st.title('Next Word Prediction with LSTM')

input_text = st.text_input("Enter the sequence of words", "To be or not to be")

if st.button("predict next word"):
    max_sequence_len = model.input_shape[1]+1 #retrieving the max sequence length from the model
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Next word: {next_word}")