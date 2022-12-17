
from modelos.bert.bert_pre_treatment_product import pre_process_text
import os
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from transformers import BertModel
from transformers import  BertTokenizer
import pickle
import torch



pre_treatment = pre_process_text(stopwords_language = 'portuguese',flg_stemm = True , flg_lemm = False)

PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


max_lenght = 38

encoders = {'segmento':pickle.load(open(os.path.join('modelos/bert/config/segmento_label_enc.dat'),'rb')),
            'categoria':pickle.load(open(os.path.join('modelos/bert/config/categoria_label_enc.dat'),'rb')),
            'subcategoria':pickle.load(open(os.path.join('modelos/bert/config/subcategoria_label_enc.dat'),'rb')),
            'nm_product':pickle.load(open(os.path.join('modelos/bert/config/nm_product_label_enc.dat'),'rb'))}


### Funcoes auxiliares ####
class ProdutosVarejoDataset(Dataset):

  def __init__(self, nm_item, tokenizer, max_len):
    self.nm_item = nm_item
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.nm_item)
  
  def __getitem__(self, item):
    
    nm_item = self.nm_item.iloc[item,0]

    nm_item2 = pre_treatment.transform(nm_item)

    encoding = self.tokenizer.encode_plus(
      nm_item2,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'nm_item_text': nm_item,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
    }

def create_data_loader(X,tokenizer, max_len, batch_size=64):
  ds = ProdutosVarejoDataset(
    nm_item = X,
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

class ProductClassifier(nn.Module):

  def __init__(self, number_classes,list_drop_out):

    super(ProductClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)

    self.drop = nn.Dropout(p= list_drop_out)

    # O last_hidden_state é uma sequência de estados ocultos da última camada do modelo
    self.out1 = nn.Linear(self.bert.config.hidden_size, number_classes[0])
    self.out2 = nn.Linear(self.bert.config.hidden_size, number_classes[1])
    self.out3 = nn.Linear(self.bert.config.hidden_size, number_classes[2])
    self.out4 = nn.Linear(self.bert.config.hidden_size, number_classes[3])
  
  def forward(self, input_ids, attention_mask):

    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)

    return [self.out1(output),self.out2(output),self.out3(output),self.out4(output)]
######




## Pipeline escoragem modelo 
def user_input_bert (text,topk = None):

    if isinstance(text, str):

        text2 =  pre_treatment.transform(text)

    else:

        test_data_loader = create_data_loader(text, tokenizer, max_lenght)
    
    model = torch.load(os.path.join('modelos/bert/config/BERT_FINAL_MODELO.pth'), map_location=torch.device('cpu'))
    model = model.to(device).eval() 

    table_final = pd.DataFrame()
    with torch.no_grad():

        if isinstance(text, str):

            encoded_review = tokenizer.encode_plus(
                        text2,
                        max_length=max_lenght,
                        add_special_tokens=True,
                        return_token_type_ids=False,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                        )

            input_ids = encoded_review['input_ids'].to(device)
            attention_mask = encoded_review['attention_mask'].to(device)

            output = model(input_ids, attention_mask)

            probs = list(map(lambda x : F.softmax(x, dim=1),output))

            table_final = pd.DataFrame()

            if not topk:

                topk = 5

            topk_cols = ['Top '+str(x) for x in range(1,topk+1)]

            for i,name in enumerate(['segmento','categoria','subcategoria','nm_product']):

                _, prediction = torch.topk(probs[i], topk)
                index = prediction.cpu().data.numpy().flatten()

                top_group = encoders[name].inverse_transform(index)

                apoio = pd.DataFrame(index=topk_cols,columns = [name],
                                    data = top_group)
                
                table_final = pd.concat([table_final,apoio],axis=1)

            
        else:
            
            final_preds1 = []
            final_preds2 = []
            final_preds3 = []
            final_preds4 = []
            nm_item_text_final = []


            for d in test_data_loader:
                
                print(len(d))
                nm_item_text = d['nm_item_text']
                input_ids_ = d["input_ids"].to(device)
                attention_mask_ = d["attention_mask"].to(device)

                output = model(input_ids_, attention_mask_)

                probs = list(map(lambda x : F.softmax(x, dim=1),output))

                nm_item_text_final.extend(nm_item_text)
                final_preds1.extend(probs[0])
                final_preds2.extend(probs[1])
                final_preds3.extend(probs[2])
                final_preds4.extend(probs[3])

            final_preds1 = torch.stack(final_preds1).cpu()
            final_preds2 = torch.stack(final_preds2).cpu()
            final_preds3 = torch.stack(final_preds3).cpu()
            final_preds4 = torch.stack(final_preds4).cpu()

            final_preds = [final_preds1,final_preds2,final_preds3,final_preds4]

            for i,name in enumerate(['segmento','categoria','subcategoria','nm_product']):

                best_values =  torch.max(final_preds[i],dim=1)[1].cpu()

                best_group = encoders[name].inverse_transform(best_values)

                apoio = pd.DataFrame(index=nm_item_text_final,columns = [name],
                                    data = best_group)
                
                table_final = pd.concat([table_final,apoio],axis=1)


        return table_final.to_html()

