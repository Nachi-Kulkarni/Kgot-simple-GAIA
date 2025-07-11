import logging
import logging.handlers
import sys
from pathlib import Path

# Define the log directory relative to this file's location
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a logger.

    Args:
        name (str): The name of the logger.
        level (int): The logging level.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    # Console formatter
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # File formatter
    file_formatter = logging.Formatter(
        '{"level": "%(levelname)s", "timestamp": "%(asctime)s", "name": "%(name)s", "message": "%(message)s"}'
    )

    # Create console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(console_formatter)
    logger.addHandler(stream_handler)

    # Create file handlers for combined and error logs
    combined_log_path = LOG_DIR / f"{name}_combined.log"
    error_log_path = LOG_DIR / f"{name}_error.log"

    # Combined log handler
    file_handler = logging.handlers.RotatingFileHandler(
        combined_log_path, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Error log handler
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_path, maxBytes=5*1024*1024, backupCount=2
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    return logger

class CustomLogger:
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = get_logger(name, level)

    def info(self, msg, extra=None):
        self.logger.info(msg, extra=self._format_extra(extra))

    def error(self, msg, extra=None):
        self.logger.error(msg, extra=self._format_extra(extra))
        
    def warn(self, msg, extra=None):
        self.logger.warning(msg, extra=self._format_extra(extra))
        
    def debug(self, msg, extra=None):
        self.logger.debug(msg, extra=self._format_extra(extra))

    def _format_extra(self, extra):
        if extra:
            return {'extra': extra}
        return None 