import os
import config
from sklearn.impute import SimpleImputer
from util import DataLoader
from logger import AppLogger

TIME_THRESHOLD = config['TIME_THRESHOLD']
TEST_SIZE = config['TEST_SIZE']
RANDOM_STATE = config['RANDOM_STATE']


class DataTransformation:
    """
    This class shall be used for handling data transformation done on the Raw Training Data.

    Written By: Hiren Kelaiya
    Version: 1.0
    Revisions: None
    """
    def __init__(self, path: str):
        self.path = path
        self.logger = AppLogger()
        self.load_data = DataLoader()
    
    def scaler(self):
        pass

    def impute_null_values(self):
        """
        Imputation of mean values in the data where null values are present
        """
        try:
            f = open('./logs/data_imputation_log.txt', 'a+')
            self.logger.log(f, 'Imputation of Missing Values Started!')

            # Load and split the data
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_data.load_and_split(TIME_THRESHOLD, TEST_SIZE, RANDOM_STATE)

            # sum(df.isnull().any(axis=1))/df.shape[0]

            imputer = SimpleImputer(strategy='mean')
            imputer.fit(X_train)

            X_train_mean_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
            X_val_mean_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
            
            self.logger.log(f, 'Imputation of Missing Values Complete!')
            f.close()

            # Save imputed data into csv
            pd.concat([X_train_mean_imputed, y_train], axis=1).to_csv('./data/train.csv').rename(columns={'time': 'outcome'})
            pd.concat([X_val_mean_imputed, y_val], axis=1).to_csv('./data/val.csv').rename(columns={'time': 'outcome'})
            pd.concat([X_test, y_test], axis=1).to_csv('./data/test.csv').rename(columns={'time': 'outcome'})
            
        except Exception as e:
            with open('./logs/data_imputation_log.txt', 'a+') as f:
                self.logger.log(f, str(e))
        
            raise e