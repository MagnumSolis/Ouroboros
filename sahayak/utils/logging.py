"""
Logging configuration using Loguru
"""

import sys
from loguru import logger

from ..config import settings


def setup_logging():
    """Configure structured logging"""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with formatting
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    
    # Add file handler for debugging
    logger.add(
        "logs/sahayak_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    
    return logger


def get_logger(name: str):
    """Get a logger with the specified name"""
    return logger.bind(name=name)
