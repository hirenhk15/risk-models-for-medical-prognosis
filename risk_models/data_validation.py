import os
import json
import pandas as pd
from risk_models.logger import AppLogger


class RawDataValidation:
    """
    This class shall be used for handling all the validation done on the Raw Training Data.

    Written By: Hiren Kelaiya
    Version: 1.0
    Revisions: None
    """
    def __init__(self, path: str):
        self.path = path
        self.schema_path = 'schema_training.json'
        self.logger = AppLogger()
    
    def read_schema(self):
        try:
            with open(self.schema_path, 'r') as f:
                _data = json.load(f)

            # Extract an information of the data
            file_name = _data['SampleFileName']
            column_names = _data['ColumnName']
            column_length = _data['NumberofColumns']

            # Add logs
            with open('./risk_models/logs/data_read_schema_log.txt.', 'a+') as f:
                message = f'Filename:: {file_name}\t NumberofColumns:: {column_length}\n'
                self.logger.log(f, message)
        
        except Exception as e:
            with open('./risk_models/logs/data_read_schema_log.txt.', 'a+') as f:
                self.logger.log(f, str(e))
                
                raise e

        return column_names, column_length

    def validate_column_length(self, column_length: int) -> None:
        try:
            f = open('./risk_models/logs/data_column_length_log.txt', 'a+')
            self.logger.log(f, 'Column Length Validation Started!')

            # Read data from file
            df = pd.read_csv(self.path)
            if not df.shape[1] == column_length:
                message = 'Invalid Column Length for the file!'
                self.logger.log(f, message)
                
                raise ValueError(message)

            self.logger.log(f, 'Column Length Validation Completed!')
            f.close()
            
        except Exception as e:
            with open('./risk_models/logs/data_column_length_log.txt', 'a+') as f:
                self.logger.log(f, str(e))
        
            raise e
        
    def validate_target_name(self, column_names: dict) -> None:
        try:
            f = open('./risk_models/logs/data_target_name_log.txt', 'a+')
            self.logger.log(f, 'Target Name Validation Started!')

            # Read data from file
            df = pd.read_csv(self.path)
            
            if not 'y' in df.columns:
                message = 'Invalid Target Name!'
                self.logger.log(f, message)
                
                raise ValueError(message)

            self.logger.log(f, 'Target Name Validation Completed!')
            f.close()
            
        except Exception as e:
            with open('./risk_models/logs/data_target_name_log.txt', 'a+') as f:
                self.logger.log(f, str(e))
        
            raise e

    def validate_null_values(self):
        """
        This function validates if any column in the csv file has all values missing.
        If all the values are missing, the data is not suitable for processing.
        """
        try:
            f = open('./risk_models/logs/data_null_values_log.txt', 'a+')
            self.logger.log(f, 'Missing Values Validation Started!')

            # Read data from file
            df = pd.read_csv(self.path)
            for i, ival in enumerate(df.isnull().sum().values):
                if ival == len(df):
                    message = f'{df.columns[i]} has all the values missing!'
                    self.logger.log(f, message)

                    raise ValueError(message)
            
            self.logger.log(f, 'Missing Values Validation Complete!')
            f.close()
        
        except Exception as e:
            with open('./risk_models/logs/data_null_values_log.txt', 'a+') as f:
                self.logger.log(f, str(e))
        
            raise e