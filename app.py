import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ner_model.h5")

model = load_model()

# Load tokenizers
@st.cache_resource
def load_tokenizers():
    with open("word_tokenizer.pkl", "rb") as f:
        word_tokenizer = pickle.load(f)
    with open("tag_tokenizer.pkl", "rb") as f:
        tag_tokenizer = pickle.load(f)
    return word_tokenizer, tag_tokenizer

word_tokenizer, tag_tokenizer = load_tokenizers()
index_to_tag = tag_tokenizer.index_word

# Set max sequence length (must match training)
MAX_LEN = model.input_shape[1]  # Ensure consistency

# Define function for Named Entity Recognition (NER)
def predict_ner(sentence):
    words = sentence.split()
    sequence = word_tokenizer.texts_to_sequences([words])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

    predictions = model.predict(padded_sequence)
    predicted_tags = np.argmax(predictions, axis=-1)

    output = [{"word": word, "tag": index_to_tag.get(tag, "O")} for word, tag in zip(words, predicted_tags[0])]
    return output

# Streamlit UI
st.title("📝 Named Entity Recognition (NER) App")
st.markdown("Enter a sentence below and see named entity predictions!")

# User Input
sentence = st.text_input("Enter a sentence:", "Michael Jackson visited New York")

if st.button("Predict"):
    if sentence.strip():
        ner_result = predict_ner(sentence)
        st.subheader("Predicted Entities:")
        for item in ner_result:
            st.write(f"**{item['word']}** → `{item['tag']}`")
    else:
        st.error("Please enter a valid sentence.")
