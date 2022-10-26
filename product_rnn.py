import streamlit as st
import numpy as np
import re
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import pickle
from PIL import Image

# ler o json
with open('product.json', 'r') as myfile:
    data=myfile.read()
produtos = json.loads(data)

label_segmento = np.sort(np.array(produtos['segmento']))
label_categoria = np.sort(np.array(produtos['categoria']))
label_subcategoria = np.sort(np.array(produtos['subcategoria']))
label_produto = np.sort(np.array(produtos['produto']))

# Abringo o Tokenizador
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# carregando o modelo
pesos = "product_rnn.h5"
model = tf.keras.models.load_model(pesos)

# Dimensão do Embbeding.
EMBEDDING_DIM = 100

# Número máximo que sequência que a rede neural irá utilizar
MAX_SEQUENCE_LENGTH = 15

# O número máximo de palavras a serem usadas. (mais frequente)
MAX_NB_WORDS = 28000

st.title('Short Text Product Classification')

img = Image.open('mercado.jpg')
st.image(img)

st.text(" ")
st.text(" ")

# CREATE ADDRESS
st.sidebar.header('User Input Features')
text = st.sidebar.text_input("Product Name", 'Macarrão')

# Função para limpar o dataset
def remove_stopwords(sentence):
  # Converting to Lowercase
  sentence = sentence.lower()

  return sentence

btn_predict = st.sidebar.button("REALIZAR PREDIÇÃO")

if btn_predict:
  new_complaint = remove_stopwords(text)
  seq = tokenizer.texts_to_sequences([new_complaint])
  padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
  pred = model.predict(padded)
  st.header(f'Segmento: {label_segmento[np.argmax(pred[0])]}')
  st.header(f'Categoria: {label_categoria[np.argsort(pred[1].flatten())[::-1]][:3]}')
  st.header(f'Subcategoria: {label_subcategoria[np.argsort(pred[2].flatten())[::-1]][:5]}')
  st.header(f'Produto: {label_produto[np.argsort(pred[3].flatten())[::-1]][:5]}')