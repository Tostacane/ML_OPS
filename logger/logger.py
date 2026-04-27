import logging
from logging.handlers import RotatingFileHandler
import structlog
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
):
    level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        rotating_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handlers.append(rotating_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)
