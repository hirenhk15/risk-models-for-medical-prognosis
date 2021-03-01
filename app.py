import os
import json
import config
from flask import Flask, request
from flask import Response, render_template
from flask_cors import CORS, cross_origin
from risk_models.model_training import TrainModel
from risk_models.train_validation import TrainValidation
from risk_models.prediction import ModelInference
#import flask_monitoringdashboard as dashboard

app = Flask(__name__)
#dashboard.bind(app)
CORS(app)


@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_patient_outcome():
    try:
        # Get the data from request
        data = request.json()

        # Initialize model inference object and load model
        model_obj = ModelInference()
        risk_model = model_obj.load_model()
        
        # Get prediction for give request parameters
        y_pred = model_obj.get_prediction(risk_model, data)

        return Response(f'10-year risk of death of a patient: {y_pred:.2f}')

    except Exception as e:
        return Response('An error occurred! %s' % e)
    
@app.route('/train', methods=['GET'])
@cross_origin()
def train_risk_model():
    try:
        path = config.RAW_DATA_PATH
        import pdb;pdb.set_trace()
        train_val_obj = TrainValidation(path)
        train_val_obj.validate()

        train_model_obj = TrainModel()
        train_model_obj.train()

    except Exception as e:
        return Response('An error occurred! %s' % e)
    
    return Response('Training successfull!')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)