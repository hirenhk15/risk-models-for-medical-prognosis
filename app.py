import os
import json
import config
from flask import Flask, request
from flask import Response, render_template
from flask_cors import CORS, cross_origin
from risk_models.model_training import TrainModel
from risk_models.train_validation import TrainValidation
#import flask_monitoringdashboard as dashboard

app = Flask(__name__)
#dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_patient_outcome():
    pass

@app.route("/train", methods=['GET'])
@cross_origin()
def train_risk_model():
    try:
        path = config['DATA_PATH']

        train_val_obj = TrainValidation(path)
        train_val_obj.validate()

        train_model_obj = TrainModel()
        train_model_obj.train()

    except Exception as e:
        return Response('An Error Occurred! %s' % e)
    
    return Response('Training successfull!')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)