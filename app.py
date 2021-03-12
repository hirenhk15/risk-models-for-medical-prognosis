import os
import json
import config
import pandas as pd
from flask import Flask, request, jsonify
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
    """
    Home page to the web app
    """
    return render_template('index.html')

@app.route('/test/data', methods=['GET'])
@cross_origin()
def load_test_Data():
    """
    This API handler is used to load test data for prediction
    """
    try:
        # Check if test.csv is present or not! (Model training is required before predictions)
        test_path = './risk_models/data/test_samples.csv'
        if not os.path.isfile(test_path):
            raise FileNotFoundError('Model training is required! Please train model first.')
        
        # Read test data
        test = pd.read_csv(test_path)

        # Select random single record from the test data
        random_data = test.sample(n=1).rename(columns={'Unnamed: 0': 'Id'}).to_dict(orient='r')
        
        return jsonify(random_data[0])

    except Exception as e:
        return Response('An error occurred! %s' % e)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_patient_outcome():
    """
    This API handler is used to make predictions on patients medical history data
    """
    try:
        # Get the data from request
        #data = request.get_json()
        data = request.form.to_dict()
        data.pop('Name')

        # Convert all the values into float
        for k,v in data.items():
            data[k] = float(v)
        
        # Initialize model inference object and load model
        model_obj = ModelInference()
        risk_model = model_obj.load_model()
        
        # Get prediction for give request parameters
        y_pred, score = model_obj.get_prediction(risk_model, data)

        return Response(f'10-year risk of death of a patient: {y_pred} (Score: {score:.2f})')

    except Exception as e:
        return Response('An error occurred! %s' % e)
    
@app.route('/train', methods=['GET'])
@cross_origin()
def train_risk_model():
    """
    This API handler used to train machine learning model on train data
    """
    try:
        path = config.RAW_DATA_PATH
        
        train_val_obj = TrainValidation(path)
        train_val_obj.validate()

        train_model_obj = TrainModel()
        train_model_obj.train()

    except Exception as e:
        return Response('An error occurred! %s' % e)
    
    return Response('Training successfull!')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)