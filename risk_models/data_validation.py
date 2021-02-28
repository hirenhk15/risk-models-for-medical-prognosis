import os
from logger import AppLogger

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