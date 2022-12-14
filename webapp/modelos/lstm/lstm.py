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

label_segmento = np.array(produtos['segmento'])
label_categoria = np.array(produtos['categoria'])
label_subcategoria = np.array(produtos['subcategoria'])
label_produto = np.array(produtos['nm_product'])

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
    segmento = label_segmento[np.argsort(pred[0].flatten())[::-1]][:5]
    categoria = label_categoria[np.argsort(pred[1].flatten())[::-1]][:5]
    subcategoria = label_subcategoria[np.argsort(pred[2].flatten())[::-1]][:5]
    produto = label_produto[np.argsort(pred[3].flatten())[::-1]][:5]

    # Criando o Data Frame
    index_labels=['Top1','Top2','Top3','Top4', 'Top5']
    labels = {
        'segmento':segmento,
        'categoria':categoria,
        'subcategoria':subcategoria,
        'produto':produto
    }
    df_product = pd.DataFrame(labels, index=index_labels)
    return df_product.to_html()

#FAZ A CATEGORIZAÇÃO DE ARQUIVOS.CSV   
def user_input_csv(dt):

    df = pd.DataFrame(
            [],
            columns=[
                'Item',
                'Segmento',
                'Categoria',
                'Subcategoria',
                'Produto',
            ],
        )

    for index, text in dt.iterrows():
        text_processed = pre_process.transform(text['nm_item'])
        seq = tokenizer.texts_to_sequences([text_processed])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)
        df = df.append(
                    {
                        'Item': text['nm_item'],
                        'Segmento': label_segmento[np.argmax(pred[0])],
                        'Categoria': label_categoria[np.argmax(pred[1])],
                        'Subcategoria': label_subcategoria[np.argmax(pred[2])],
                        'Produto': label_produto[np.argmax(pred[3])],
                    },
                    ignore_index=True,
                )

    return df.to_html()
