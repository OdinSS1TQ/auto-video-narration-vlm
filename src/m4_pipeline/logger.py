"""
Logger — Structured logging configuration using loguru.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    colorize: bool = True,
):
    """
    Configure loguru logger for the pipeline.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.
        colorize: Whether to colorize console output.
    """
    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=log_level,
        colorize=colorize,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{module}:{function}:{line} | "
                "{message}"
            ),
        )

    logger.info(f"Logger initialized (level={log_level})")


def get_logger(name: str = None):
    """
    Get a contextualized logger.

    Args:
        name: Logger context name (e.g., module name).

    Returns:
        Loguru logger instance.
    """
    if name:
        return logger.bind(context=name)
    return logger
