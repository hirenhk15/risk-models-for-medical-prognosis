from datetime import datetime

class AppLogger:
    """Logging every process in this app
    """
    def __init__(self):
        pass
    
    def log(self, file_object: object, log_message: str) -> None:
        """AI is creating summary for log

        Args:
            file_object (object): A file object
            log_message (str): Log message of the process
        """
        self.now = datetime.utcnow()
        self.date = self.now.date()
        self.current_time = self.now.strftime('%H:%M:%S')

        file_object.write(
            str(self.date) + '/' + str(self.current_time) + '\t\t' + log_message + '\n'
        )