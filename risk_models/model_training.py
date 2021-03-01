import os
import config
import pandas as pd
from risk_models.util import cindex
from risk_models.logger import AppLogger
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# Load model and hyperparameters
MODEL = config.MODEL
HYPERPARAMS = config.HYPERPARAMS
CROSS_VALIDATION = config.CROSS_VALIDATION


class TrainModel:
    """
    This class shall be used train risk models on training data.

    Written By: Hiren Kelaiya
    Version: 1.0
    Revisions: None
    """
    def __init__(self):
        """
        Initialize TrainModel class arguments
        """
        self.logger = AppLogger()
        self.file_object = open('./risk_models/logs/model_training_log.txt', 'a+')

    def _get_data(self):
        self.train_df = pd.read_csv('./risk_models/data/train.csv')
        self.val_df = pd.read_csv('./risk_models/data/val.csv')
        self.test_df = pd.read_csv('./risk_models/data/test.csv')

    def _feature_target_split(self):
        self.X_train = self.train_df.drop('outcome', axis=1)
        self.y_train = self.train_df['outcome']

        self.X_val = self.val_df.drop('outcome', axis=1)
        self.y_val = self.val_df['outcome']

        self.X_test = self.test_df.drop('outcome', axis=1)
        self.y_test = self.test_df['outcome']

    def _c_index(self):
        return make_scorer(cindex, greater_is_better=False)
    
    def _save_model(self, model):
        self.logger.log(self.file_object, 'Saving Model...')

        try:
            # Create model directory if not present
            _model_dir = './risk_models/model'
            if os.path.isdir(_model_dir):
                os.makedirs(_model_dir)

            # Open a file, where you ant to store the data
            with open('./risk_models/model/risk_model.pkl', 'wb') as f:
                # Dump information to that file
                pickle.dump(model, f)

            self.logger.log(self.file_object, 'Model Saved Successfully!')
        
        except Exception as e:
            with self.file_object as f:
                self.logger.log(f, 'Failed to Save Model!')
        
            raise e
        
    def train(self):
        # Add logs when training starts
        self.logger.log(self.file_object, 'Start of Training!')

        try:
            # Get the data for training and split into train, val and test
            self._feature_target_split()

            # Grid search for best model parameters
            grid = GridSearchCV(
                MODEL, param_grid=HYPERPARAMS, cv=CROSS_VALIDATION, verbose=1, n_jobs=-1, scoring=self._c_index
                )
            grid.fit(self.X_train, self.y_train)

            y_train_pred = grid.best_estimator_.predict_proba(X_train)[:, 1]
            y_val_pred = grid.best_estimator_.predict_proba(X_val)[:, 1]
            y_test_pred = grid.best_estimator_.predict_proba(X_test)[:, 1]

            # Log c-index scores for train, val and test dataset
            message = f'C-Index Scores:: Train C-Index: {cindex(y_train, y_train_pred):.4f} | ' + \
                f'Validation C-Index: {cindex(y_val, y_val_pred):.4f} | ' + \
                f'Test C-Index: {cindex(y_test, y_test_pred):.4f}'
            
            self.logger.log(self.file_object, message)

            # Save the model into model directory
            self._save_model(grid.best_estimator_)
            
            # Logging the successful Training
            self.logger.log(self.file_object, 'Successful End of Training')
            self.file_object.close()
        
        except Exception as e:
            message = 'Unsuccessful End of Training'
            with self.file_object as f:
                self.logger.log(f, message)
        
            raise e