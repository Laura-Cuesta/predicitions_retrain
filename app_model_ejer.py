# from crypt import methods
from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods =['GET'])
def hello ():
    return "Bienvenido a mi API del modelo advertising"

@app.route('/api/v1/predict', methods=['GET'])# creamos la ruta api para obtener los dtos
def predict():
    
    try:
        model = pickle.load(open('model/ad_model.pkl','rb'))# leemos el modelo
        tv = request.args.get('tv', None)# cogete tv, si no está ponle none
        radio = request.args.get('radio', None)# cogete radio, si no está ponle none
        newspaper = request.args.get('newspaper', None)# cogete newspapel, si no está ponle none

        if tv is None or radio is None or newspaper is None:
            return "Args empty, the data are not enough to predict"
        else:
            prediction = model.predict([[tv,radio,newspaper]])
        
        return jsonify({'predictions': prediction[0]})
    except:
        return 'No hay ningún modelo entrenado, necesitamos entrenar primero el modelo para poder hacer las predicciones'


@app.route('/api/v1/retrain', methods=['GET','POST','PUT'])

def retrain ():
    
      
    
    data = pd.read_csv('data/Advertising.csv', index_col=0)

    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

    model = Lasso(alpha=6000)
    model.fit(X_train, y_train)

    pickle.dump(model, open('model/ad_model.pkl', 'wb'))

    
    return str(np.sqrt(mean_squared_error(y_test, model.predict(X_test))) )


app.run()


# pip install -i https://test.pypi.org/simple/ omarpy