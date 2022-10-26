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

# Abrindo o Tokenizador
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# carregando o modelo
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("product_rnn.h5")
    return model
    
model = load_model()

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

# Texto do item
st.sidebar.header('Entrada do Texto')
text = st.sidebar.text_input("NOME DO ITEM", 'Biscoito de Chocolate')

# Função para limpar o dataset
def remove_stopwords(sentence):
  # Converting to Lowercase
  sentence = sentence.lower()

  return sentence

btn_predict = st.sidebar.button("REALIZAR CASSIFICAÇÃO")

if btn_predict:
  new_complaint = remove_stopwords(text)
  seq = tokenizer.texts_to_sequences([new_complaint])
  padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
  pred = model.predict(padded)
  st.header(f'Segmento:')
  st.text(label_segmento[np.argmax(pred[0])])
  st.header('Categoria:')
  st.text(label_categoria[np.argsort(pred[1].flatten())[::-1]][:3])
  st.header('Subcategoria:')
  st.text(label_subcategoria[np.argsort(pred[2].flatten())[::-1]][:3])
  st.header('Produto:')
  st.text(label_produto[np.argsort(pred[3].flatten())[::-1]][:5])