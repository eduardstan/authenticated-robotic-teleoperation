import logging
import sys
import os

def get_logger(name, log_file=None, level=logging.INFO):
    """
    Creates and configures a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): File path to save the log file. If None, logs are not saved to a file.
        level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers are already added to avoid duplicate logs
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)s in %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file:
            # Ensure the log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
