# Structured logging
import logging

logger = logging.getLogger("proshop_ai")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

def log_info(message: str):
    logger.info(message)

def log_error(message: str):
    logger.error(message)
