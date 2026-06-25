import logging
from rich.logging import RichHandler

def get_logger(name: str) -> logging.Logger:
    """
    Mengembalikan instance logger dengan format Rich untuk output terminal yang lebih rapi.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Integrasi RichHandler
        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False
        )
        logger.addHandler(rich_handler)
        
    return logger
