import logging
from pathlib import Path
from eepy_car.alert import AlertLevel


def setup_logger(log_path: str | Path) -> logging.Logger:
    """Sets up and returns a logger that writes alert events to file.

    Args:
        log_path (str | Path): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("eepy_car")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_path)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def log_alert(logger: logging.Logger, alert_level: AlertLevel) -> None:
    """Logs an alert level change event

    Args:
        logger (logging.Logger): The logger instance.
        alert_level (AlertLevel): The new alert level.
    """
    if alert_level == AlertLevel.NONE:
        logger.info("State returned to NORMAL")
    elif alert_level in (AlertLevel.DROWSINESS_WARNING, AlertLevel.DISTRACTION_WARNING):
        logger.warning(f"ALERT: {alert_level.name}")
    else:
        logger.critical(f"ALERT: {alert_level.name}")
