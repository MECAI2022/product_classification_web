from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})
    