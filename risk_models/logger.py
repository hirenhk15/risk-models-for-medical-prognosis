import os
from datetime import datetime

class AppLogger:
    """Logging every process in this app
    """
    def __init__(self):
        pass

    def create_log_dir(self):
        """
        Create logs directory if not present
        """
        _log_dir = './risk_models/logs'
        if not os.path.isdir(_log_dir):
            os.makedirs(_log_dir)
        
    def log(self, file_object: object, log_message: str) -> None:
        """Write log files

        Args:
            file_object (object): A file object
            log_message (str): Log message of the process
        """
        # Create logs dir if not present
        self.create_log_dir()

        self.now = datetime.utcnow()
        self.date = self.now.date()
        self.current_time = self.now.strftime('%H:%M:%S')

        file_object.write(
            str(self.date) + '/' + str(self.current_time) + '\t\t' + log_message + '\n'
        )