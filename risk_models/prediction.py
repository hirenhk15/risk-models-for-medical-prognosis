# This module is for model inference tasks

import os
import config
import pickle
from logger import AppLogger


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
    
    def load_model(self):
        """
        Load saved machine learning model to make predictions.
        """
        self.logger.log(self.file_object, 'Loading Model...')

        try:
            # Load model from model directory
            with open('./model/best_model.pkl', 'rb') as f:
                best_model = pickle.load(f)

            self.logger.log(self.file_object, 'Model Loaded Successfully!')

            return best_model

        except Exception as e:
            with self.file_object as f:
                self.logger.log(f, 'Failed to Load Model! The Reason might be Model not Saved or Training is Pending!')
        
            raise e

    def get_prediction(self, model: object, data: pd.Series):
        return model.predict_proba(data)[:, 1]

    def feature_importance(self):
        pass