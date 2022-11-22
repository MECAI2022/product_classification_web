from flask import Flask
app = Flask (__name__)

@app.route('/')
def home():
    return "<h1>Short Classification/h1>"
app.run()

