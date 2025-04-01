from flask import Flask, redirect, url_for,render_template
from flask_cors import CORS
from space_identification import space_identification_bp  
from Disease_Prediction import disease_identification_bp
from plant_prediction import plant_prediction_bp
from Chatbot.Backend.blueprints.chatbot import chatbot_bp
from Chatbot.Backend.blueprints.plant import plant_bp
import subprocess

app = Flask(__name__)
app.secret_key = "EcoGrow" 
CORS(app)

app.register_blueprint(space_identification_bp, url_prefix='/space_identification')
app.register_blueprint(disease_identification_bp, url_prefix = '/disease_identification')
app.register_blueprint(plant_prediction_bp, url_prefix = '/plant_prediction')
app.register_blueprint(chatbot_bp,  url_prefix = '/chatbot')
app.register_blueprint(plant_bp, url_prefix='/plant')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat')
def redirect_to_chatbot():
    return redirect('http://127.0.0.1:5000/')

if __name__ == '__main__':
    app.run(debug=True,port=5001)


