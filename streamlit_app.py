# pip install streamlit gdown
import os
import gdown

import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from transformer import Transformer
from keras.models import load_model
from keras.saving import register_keras_serializable
import string
import re
import streamlit as st

# # 🔹 Replace these with your actual Google Drive file IDs
WEIGHTS_FILE_ID = "1ro0_qn_UKpZIPszln_CQ6PFHjkqLdwNw"
SOURCE_VEC_ID   = "1pLVLE46Tle94LuA7wGEjctEA1FIu41rO"
TARGET_VEC_ID   = "1A14nuEK3KDuJ6bGC137nKVox0HZSCJHe"

# # 🔻 Function to download from Google Drive
def download_file_from_drive(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# # 🔹 Download model and vectorizers if not already present
download_file_from_drive(WEIGHTS_FILE_ID, "spa_translation_transformer.weights.h5")
download_file_from_drive(SOURCE_VEC_ID, "source_vectorization.keras")
download_file_from_drive(TARGET_VEC_ID, "target_vectorization.keras")

# 🔹 Register custom standardization
@register_keras_serializable()
def custom_standardization(input_string):
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "").replace("]", "")
    return tf.strings.regex_replace(tf.strings.lower(input_string), f"[{re.escape(strip_chars)}]", "")

# 🔹 Load vectorizers
source_vectorization = load_model("source_vectorization.keras").layers[1]
target_vectorization = load_model("target_vectorization.keras").layers[1]

# 🔹 Vocab for decoding
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

# 🔹 Rebuild and load model
vocab_size = 15000
seq_length = 20
model = Transformer(n_layers=4, d_emb=128, n_heads=8, d_ff=512,
                    dropout_rate=0.1,
                    src_vocab_size=vocab_size,
                    tgt_vocab_size=vocab_size)

# Build model once
example_sentence = "hello"
src = source_vectorization([example_sentence])
tgt = target_vectorization(["[start] hello [end]"])[:, :-1]
model((src, tgt))  # build model with real input shapes

model.load_weights("spa_translation_transformer.weights.h5")

# 🔹 Translation function
def translate(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(seq_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = model((tokenized_input_sentence, tokenized_target_sentence))
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence.replace("[start] ", "").replace(" [end]", "")

# 🔹 Streamlit UI
st.title("English to Spanish Translation")
st.write("Enter an English sentence below:")

user_input = st.text_input("Your English sentence:")

if user_input:
    with st.spinner("Translating..."):
        translation = translate(user_input)
    st.success(f"Spanish: {translation}")
