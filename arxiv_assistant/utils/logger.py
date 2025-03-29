import logging
import os
from datetime import datetime
import traceback

# Define log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Get current date for log file name
current_date = datetime.now().strftime("%Y-%m-%d")
ERROR_LOG_FILE = os.path.join(LOG_DIR, f"errors_{current_date}.log")
SEARCH_LOG_FILE = os.path.join(LOG_DIR, f"search_{current_date}.log")
SUMMARY_LOG_FILE = os.path.join(LOG_DIR, f"summary_{current_date}.log")

# Configure root logger
def configure_root_logger():
    """Configure the root logger with console and file handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Create error file handler
    error_file_handler = logging.FileHandler(ERROR_LOG_FILE)
    error_file_handler.setLevel(logging.ERROR)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    error_file_handler.setFormatter(file_format)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(error_file_handler)
    
    return root_logger

# Get or create logger
def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)

# Log exception with traceback
def log_exception(logger, message, exception):
    """Log an exception with full traceback."""
    error_msg = f"{message}: {str(exception)}"
    tb = traceback.format_exc()
    logger.error(f"{error_msg}\n{tb}")
    
    # Also write to the error log file directly to ensure it's captured
    with open(ERROR_LOG_FILE, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{timestamp}] {error_msg}\n{tb}\n")
    
    return error_msg

# Configure search logger
def get_search_logger():
    """Get a logger specifically for search operations."""
    logger = logging.getLogger("search")
    logger.setLevel(logging.INFO)
    
    # Create file handler for search logs
    search_file_handler = logging.FileHandler(SEARCH_LOG_FILE)
    search_file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    search_file_handler.setFormatter(file_format)
    
    # Add handler if not already added
    if not logger.handlers:
        logger.addHandler(search_file_handler)
    
    return logger

# Configure summary logger
def get_summary_logger():
    """Get a logger specifically for paper summarization operations."""
    logger = logging.getLogger("summary")
    logger.setLevel(logging.INFO)
    
    # Create file handler for summary logs
    summary_file_handler = logging.FileHandler(SUMMARY_LOG_FILE)
    summary_file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    summary_file_handler.setFormatter(file_format)
    
    # Add handler if not already added
    if not logger.handlers:
        logger.addHandler(summary_file_handler)
    
    return logger

# Initialize root logger
configure_root_logger() 