from flask import Flask, render_template, request,session
from werkzeug.utils import secure_filename
import pandas as pd
import os
from modelos.lstm.lstm import user_input
from modelos.lstm.lstm import user_input_csv
from modelos.bert.bert_pipeline import user_input_bert
from modelos.bert.bert_pipeline import ProductClassifier 



app = Flask(__name__)

@app.route('/')
def inicio():
    return render_template('index.html');

@app.route('/modelo', methods=['GET', 'POST'])
def modelo():
        # handle the POST request
    if request.method == 'POST':
        mytext = request.form.get('mytext')
        mymodel = request.form.get('mymodel')
        
        if mymodel == 'Bert':
            #Aqui vai a função do modelo 1
            mytext = user_input_bert(mytext)
            return  mytext
            
            
        if mymodel == 'LSTM':
            #Aqui vai a função do modelo 2
            mytext = user_input(mytext)
            return  mytext
           
        return 0    
            

         
   



# Faz o Upload do CSV
@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    mymodel = request.form.get('mymodel')
    if 'csvfile' in request.files:
        file = request.files['csvfile']
        if file.filename == '':
            return '', 400
        dest = os.path.join(
            app.instance_path,
            app.config.get('UPLOAD_FOLDER', 'files'),
            secure_filename(file.filename)
        )
        file.save(dest)
        
        dt = pd.read_csv(dest, names=['nm_item'])
        
      
        output = user_input_csv(dt)
     
        
        
        return '''{}'''.format(output), 201
    return '', 400
   


app.run(host="0.0.0.0",port=5000)