# This module is for data transformation methods

import os
import config
import pandas as pd
from sklearn.impute import SimpleImputer
from risk_models.util import DataLoader
from risk_models.logger import AppLogger

# Load model configuration parameters from the config
TIME_THRESHOLD = config.TIME_THRESHOLD
TEST_SIZE = config.TEST_SIZE
RANDOM_STATE = config.RANDOM_STATE


class DataTransformation:
    """
    This class shall be used for handling data transformation done on the Raw Training Data.

    Written By: Hiren Kelaiya
    Version: 1.0
    Revisions: None
    """
    def __init__(self, path: str):
        """Initialize model arguments

        Args:
            path (str): path of the data
        """
        self.path = path
        self.logger = AppLogger()
        self.load_data = DataLoader()
    
    def scaler(self):
        """
        Scaling function to scale the training data
        """
        pass

    def _get_data(self):
        """
        Method to load and split the data into training, validation and test sets
        """
        # Load and split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data.load_and_split(TIME_THRESHOLD, TEST_SIZE, RANDOM_STATE)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def is_null_present(self):
        """
        Method to check whether null values present in the data!
        """
        # Load and split the data
        self._get_data()
        self.null_present = False

        if self.X_train.isnull().sum().sum() != 0:
            self.null_present = True
        
        return self.null_present

    def impute_null_values(self):
        """
        Imputation of mean values in the data where null values are present
        """
        try:
            f = open('./risk_models/logs/data_imputation_log.txt', 'a+')
            self.logger.log(f, 'Imputation of Missing Values Started!')

            # # Load and split the data
            # self._get_data()
            
            # sum(df.isnull().any(axis=1))/df.shape[0]

            imputer = SimpleImputer(strategy='mean')
            imputer.fit(self.X_train)

            X_train_mean_imputed = pd.DataFrame(imputer.transform(self.X_train), columns=self.X_train.columns)
            X_val_mean_imputed = pd.DataFrame(imputer.transform(self.X_val), columns=self.X_val.columns)
            
            self.logger.log(f, 'Imputation of Missing Values Complete!')
            f.close()
            
            # Save imputed data into csv
            pd.concat([X_train_mean_imputed, self.y_train.reset_index(drop=True)], axis=1).rename(columns={'time': 'outcome'}).to_csv('./risk_models/data/train.csv')
            pd.concat([X_val_mean_imputed, self.y_val.reset_index(drop=True)], axis=1).rename(columns={'time': 'outcome'}).to_csv('./risk_models/data/val.csv')
            pd.concat([self.X_test, self.y_test.reset_index(drop=True)], axis=1).rename(columns={'time': 'outcome'}).to_csv('./risk_models/data/test.csv')
            
        except Exception as e:
            with open('./risk_models/logs/data_imputation_log.txt', 'a+') as f:
                self.logger.log(f, str(e))
        
            raise e