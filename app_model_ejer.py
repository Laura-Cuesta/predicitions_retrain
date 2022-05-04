from flask import Flask, jsonify, request
import os
import pickle

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True



@app.route('/api/v1/predict', methods=['GET'])# creamos la ruta api para obtener los dtos

def predict():
    
    model = pickle.load(open('model/ad_model.pkl','rb'))# leemos el modelo
    tv = int (input ('introduce el importe para tv: '))
    radio = int (input ('introduce el importe para radio: '))
    newspaper = int (input ('introduce el importe para newspaper: '))
    # tv = request.args.get('tv', None)# cogete tv, si no está ponle none
    # radio = request.args.get('radio', None)# cogete radio, si no está ponle none
    # newspaper = request.args.get('newspaper', None)# cogete newspapel, si no está ponle none

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
    
    return jsonify({'predictions': prediction[0]})

app.run()