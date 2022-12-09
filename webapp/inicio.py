from flask import Flask, render_template, request,session
from werkzeug.utils import secure_filename
import pandas as pd
import os
from modelos.lstm.lstm import user_input


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
            mytext = user_input(mytext)
            
            

         
        return '''
                  <h2>Classificação</h2>
<p>Modelos escolhido:{}</p>
<table>
  <tr>
    <th>Segmento</th>
    <th>Categoria</th>
    <th>Subcategoria</th>
    <th>Produto</th>
  </tr>
  <tr>
    <td><p>{}</p></td>
    <td><p>{}</p></td>
    <td><p>{}</p></td>
    <td><p>{}</p></td>
  </tr>
  
  
</table> '''.format(mymodel,mytext[0],mytext[1],mytext[2],mytext[3] )



# Faz o Upload do CSV
@app.route('/upload-csv', methods=['POST'])
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
        df = pd.read_csv(dest)
        file.save(dest)
        return '''{}'''.format(df), 201
    return '', 400
   


app.run()