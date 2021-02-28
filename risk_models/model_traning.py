import os
from util import cindex
from logger import AppLogger

class TrainModel:
    """
    This class shall be used train risk models on training data.

    Written By: Hiren Kelaiya
    Version: 1.0
    Revisions: None
    """
    def __init__(self, path: str):
        self.path = path
        self.logger = AppLogger()
        
    def train(self):
        pass