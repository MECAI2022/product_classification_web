from flask import Flask, render_template, request,session
from werkzeug.utils import secure_filename
import pandas as pd
import os


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
            mytext = mytext.upper()
            
        if mymodel == 'LSTM':
            #Aqui vai a função do modelo 2
            mytext = mytext.lower()
            

         
        return '''
                  <h1>Resultado: {}</h1><p>modelo escolhido: {}<p>'''.format(mytext, mymodel)


# Faz o Upload do CSV
@app.route('/upload-csv', methods=['GET', 'POST'])
def upload_csv():
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
        return '', 201
    return '', 400


app.run()