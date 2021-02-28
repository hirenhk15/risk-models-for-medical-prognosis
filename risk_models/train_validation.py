from logger import AppLogger
from data_validation import RawDataValidation
from data_transform import DataTransformation


class TrainValidation:
    """
    Validation methods for training risk models
    """
    def __init__(self, path: str):
        self.path = path
        self.logger = AppLogger()
        self.raw_data = RawDataValidation(self.path)
        self.data_transform = DataTransformation(self.path)
        self.file_object = open('./logs/training_main_log.txt', 'a+')

    def validate(self):
        """Validation steps are performed in this method.

        Raises:
            e: Exceptions are raised if any
        """
        try:
            self.logger.log(self.file_object, 'Starting Data Validation!')

            # Extracting values from prediction schema
            column_names, column_length = self.raw_data.read_schema()

            # Validating column length in the file
            self.raw_data.validate_column_length(column_length)

            # Validate if target has 'outcome' or not! (also validate data types)
            self.raw_data.validate_target_name(column_names)
            
            # Validating if any column has all values missing
            self.raw_data.validate_null_values()
            self.logger.log(self.file_object, 'Raw Data Validation Complete!')
            self.logger.log(self.file_object, 'Starting Data Transforamtion!')

            # Impute missing null values
            self.data_transform.impute_null_values()

            self.log_writer.log(self.file_object, 'Data Transformation Completed!')
            self.file_object.close()
        
        except Exception as e:
            with self.file_object as f:
                self.logger.log(f, str(e))
            raise e