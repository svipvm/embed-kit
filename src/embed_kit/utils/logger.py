import logging
import sys
from logging.handlers import RotatingFileHandler


class LoggerManager:
    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    MAX_BYTES = 50 * 1024 * 1024
    BACKUP_COUNT = 5

    def __init__(self) -> None:
        self._initialized = False

    def setup(self, log_file: str | None = None) -> None:
        if self._initialized:
            return

        handlers: list[logging.Handler] = []

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(self.LOG_FORMAT, datefmt=self.DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

        if log_file:
            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=self.MAX_BYTES,
                backupCount=self.BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(self.LOG_FORMAT, datefmt=self.DATE_FORMAT)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)

        logging.basicConfig(
            level=logging.DEBUG,
            format=self.LOG_FORMAT,
            datefmt=self.DATE_FORMAT,
            handlers=handlers,
        )

        self._initialized = True

    def get_logger(self, name: str) -> logging.Logger:
        if not self._initialized:
            self.setup()
        return logging.getLogger(name)


logger_manager = LoggerManager()
get_logger = logger_manager.get_logger
