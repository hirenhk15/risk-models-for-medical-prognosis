# This module is for model inference tasks

import os
import config
import pickle
import pandas as pd
from risk_models.logger import AppLogger


class ModelInference:
    """
    This class shall be used for load model and make predictions.

    Written By: Hiren Kelaiya
    Version: 1.0
    Revisions: None
    """
    def __init__(self):
        """
        Initialize ModelInference arguments
        """
        self.logger = AppLogger()
        self.file_object = open('./risk_models/logs/model_inference_log.txt', 'a+')
    
    def load_model(self):
        """
        Load saved machine learning model to make predictions.
        """
        self.logger.log(self.file_object, 'Loading Model...')

        try:
            # Load model from model directory
            with open('./risk_models/model/risk_model.pkl', 'rb') as f:
                risk_model = pickle.load(f)

            self.logger.log(self.file_object, 'Model Loaded Successfully!')
            self.file_object.close()

            return risk_model

        except Exception as e:
            with self.file_object as f:
                self.logger.log(f, 'Failed to Load Model! The Reason might be Model not Saved or Training is Pending!')
        
            raise e

    def get_prediction(self, model: object, data: dict):
        """Get predictions from requested data

        Args:
            model (object): Trained model for inference
            data (dict): Patient medical history data

        Returns:
            y_pred (str): Yes/No for 10 year risk of death
            score (float): Risk score of patient for death
        """
        # Return the probability score (i.e. risk score) of positive class
        score = model.predict_proba(pd.DataFrame([data]))[:, 1][0]
        
        # Compare score with threshold for predictions
        if score > config.RISK_THRESHOLD:
            y_pred = 'Yes'
        else:
            y_pred = 'No'

        return y_pred, score

    def feature_importance(self):
        """
        Using SHAP values, deriving important features that contributing patients death
        """
        pass