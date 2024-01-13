from flask import Flask
from chatbot import *

# import to run the app using ngrok
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request
  
app = Flask(__name__) 
run_with_ngrok(app)
  
@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(get_response(userText))
  
if __name__ == "__main__":
    app.run()