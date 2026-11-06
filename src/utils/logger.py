import logging
import sys

from src.utils.config import get_project_root, load_config


def get_logger(name: str = "customer_behavior_platform") -> logging.Logger:
    """
    Creates and configures a standardized logger based on project configuration.

    Args:
        name: Name of the logger module.

    Returns:
        Configured logging.Logger instance.
    """
    try:
        config = load_config()
        log_level_str = config.get("logging", {}).get("level", "INFO")
        log_format = config.get("logging", {}).get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_file_rel = config.get("logging", {}).get("file", "logs/platform.log")
    except Exception:
        log_level_str = "INFO"
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_file_rel = "logs/platform.log"

    level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(log_format)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    root = get_project_root()
    log_file = root / log_file_rel
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
