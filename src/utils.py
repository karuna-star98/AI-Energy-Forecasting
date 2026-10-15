"""utils.py - small helpers shared by several modules."""
import logging
import sys

from src import config


def get_logger(name: str = "energy") -> logging.Logger:
    """Logger that prints to the console AND writes to outputs/training_log.txt."""
    logger = logging.getLogger(name)
    if logger.handlers:                      # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    config.ensure_dirs()
    log_file = logging.FileHandler(config.OUTPUTS_DIR / "training_log.txt", mode="w")
    log_file.setFormatter(fmt)
    logger.addHandler(log_file)
    return logger
