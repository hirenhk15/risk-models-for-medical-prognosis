from logger import AppLogger

class TrainValidation:
    """Validation methods for training risk models
    """
    def __init__(self, path: str):
        self.path = path
        self.logger = AppLogger()