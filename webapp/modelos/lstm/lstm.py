import numpy as np
import pandas as pd
import json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
import tensorflow_addons as tfa
from modelos.lstm.pre_treatment_product import pre_process_text
import pickle


pre_process = pre_process_text(stopwords_language='portuguese')
# ler as categorias
with open('modelos/lstm/product.json', 'r') as myfile:
    data=myfile.read()
produtos = json.loads(data)

label_segmento = np.sort(np.array(produtos['segmento']))
label_categoria = np.sort(np.array(produtos['categoria']))
label_subcategoria = np.sort(np.array(produtos['subcategoria']))
label_produto = np.sort(np.array(produtos['nm_product']))

# Abrindo o Tokenizador
with open('modelos/lstm/tokenizer_last.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# carregando o modelo
def load_model():
    model = tf.keras.models.load_model("modelos/lstm/pesos/full_MultiModel2.h5")
    return model
    
model = load_model()
MAX_SEQUENCE_LENGTH = 15

#FAZ A CATEGORIZAÇÃO PARA TEXTO SIMPLES
def user_input(usertext):
    text = usertext
    new_complaint = pre_process.transform(text)
    seq = tokenizer.texts_to_sequences([new_complaint])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    segmento = f'{label_segmento[np.argmax(pred[0])]}'
    categoria = f'{label_categoria[np.argsort(pred[1].flatten())[::-1]][:3]}'
    subcategoria = f'{label_subcategoria[np.argsort(pred[2].flatten())[::-1]][:3]}'
    produto = f'{label_produto[np.argsort(pred[3].flatten())[::-1]][:5]}'
    return segmento,categoria,subcategoria,produto

#FAZ A CATEGORIZAÇÃO DE ARQUIVOS.CSV   
def user_input_csv(dt):

    df = pd.DataFrame(
            [],
            columns=[
                'nm_item',
                'segmento',
                'categoria',
                'subcategoria',
                'nm_product',
            ],
        )

    for index, text in dt.iterrows():
        text_processed = pre_process.transform(text['nm_item'])
        seq = tokenizer.texts_to_sequences([text_processed])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)
        df = df.append(
                    {
                        'nm_item': text['nm_item'],
                        'segmento': label_segmento[np.argmax(pred[0])],
                        'categoria': label_categoria[np.argmax(pred[1])],
                        'subcategoria': label_subcategoria[np.argmax(pred[2])],
                        'nm_product': label_produto[np.argmax(pred[3])],
                    },
                    ignore_index=True,
                )

    return df.to_html()
