import logging
import sys

from app.config import Settings
from app.middleware import request_id_context


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_context.get()
        return True


def configure_logging(settings: Settings) -> None:
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "request_id=%(request_id)s | "
            "%(message)s"
        )
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIDFilter())

    root_logger = logging.getLogger()

    root_logger.setLevel(settings.log_level)

    root_logger.handlers.clear()

    root_logger.addHandler(handler)

    for name in (
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "openai",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
