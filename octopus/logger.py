"""Logger."""

import io
import logging
import sys

from upath import UPath

from .types import LogGroup


class ContextualFilter(logging.Filter):
    """Contextual filer."""

    def __init__(self):
        super().__init__()
        self.current_group = LogGroup.DEFAULT
        self.process_id = None

    def filter(self, record):
        """Add group and process ID to the log record."""
        record.group = self.current_group.name
        record.process_id = self.process_id
        return True


class ExternalHandler(logging.Handler):
    """Handler that forwards external library logs to the Octopus logger."""

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def emit(self, record):
        """Forward the log record to the Octopus logger."""
        self.logger.log(record.levelno, record.getMessage())


# Define ANSI color codes
class LogColors:
    """Log Colors."""

    RESET = "\033[0m"
    INFO = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    CRITICAL = "\033[41m"  # Red background
    DEBUG = "\033[94m"  # Blue


class CustomFormatter(logging.Formatter):
    """Custom Formatter."""

    def format(self, record):
        """Create custom logger format."""
        if record.process_id is None:
            self._style._fmt = "%(asctime)s | %(levelname)s | %(group)s | %(message)s"
        else:
            self._style._fmt = "%(asctime)s | %(levelname)s | %(group)s | %(process_id)s | %(message)s"
        return super().format(record)


class FSSpecFileHandler(logging.StreamHandler):
    """A handler class which writes formatted logging records to disk files.

    The handler just opens the specified file and uses it as the stream for logging.
    """

    def __init__(self, filename: UPath, mode=None, encoding=None, errors=None):
        self.filename = filename.resolve()
        self.mode = mode or ("a" if self.filename.exists() else "w")
        self.encoding = encoding
        if "b" not in self.mode:
            self.encoding = io.text_encoding(encoding)
        self.errors = errors

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        logfile = self.filename.open(self.mode, encoding=encoding, errors=errors)
        logging.StreamHandler.__init__(self, logfile)

    def close(self):
        """Closes the stream."""
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                logging.StreamHandler.close(self)
        finally:
            self.release()

    def __repr__(self):
        level = logging.getLevelName(self.level)
        return f"<{self.__class__.__name__} {self.filename} ({level})>"


def setup_logger(name="OctoManager", log_file: UPath | None = None, level=logging.INFO):
    """Set up a logger with a file handler and a console handler."""
    # Clear existing loggers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Disable propagation for all existing loggers
    for log_name in logging.root.manager.loggerDict:
        logging.getLogger(log_name).propagate = False

    # Add the contextual filter
    contextual_filter = ContextualFilter()
    logger.addFilter(contextual_filter)

    # Create a separate formatter for console without datetime
    console_formatter = logging.Formatter("%(levelname)s | %(group)s | %(message)s")

    # Create file handler
    set_logger_filename(logger, log_file)

    # Create console handler with the new formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Route external library logs through the Octopus logger
    external_handler = ExternalHandler(logger)

    optuna_logger = logging.getLogger("optuna")
    optuna_logger.handlers.clear()
    optuna_logger.setLevel(level)
    optuna_logger.propagate = False
    optuna_logger.addHandler(external_handler)

    autogluon_logger = logging.getLogger("autogluon")
    autogluon_logger.handlers.clear()
    autogluon_logger.setLevel(level)
    autogluon_logger.propagate = False
    autogluon_logger.addHandler(external_handler)

    # Override the default log level colors
    def colorize_log(record):
        level_color = LogColors.RESET
        if record.levelname == "INFO":
            level_color = LogColors.INFO
        elif record.levelname == "WARNING":
            level_color = LogColors.WARNING
        elif record.levelname == "ERROR":
            level_color = LogColors.ERROR
        elif record.levelname == "CRITICAL":
            level_color = LogColors.CRITICAL
        elif record.levelname == "DEBUG":
            level_color = LogColors.DEBUG

        record.levelname = f"{level_color}{record.levelname}{LogColors.RESET}"
        return True

    # Add the colorizing filter to the console handler
    console_handler.addFilter(colorize_log)

    # Add method to set current group and process ID
    def set_log_group(group, process_id=None):
        if isinstance(group, LogGroup):
            contextual_filter.current_group = group
            contextual_filter.process_id = process_id
        else:
            raise ValueError("Group must be an instance of LogGroup")

    logger.set_log_group = set_log_group  # type: ignore[attr-defined]

    return logger


def set_logger_filename(logger: logging.Logger | None = None, log_file: UPath | None = None, level: int | None = None):
    """Set logger filename or disable file logging."""
    if logger is None:
        logger = get_logger()

    file_handlers = [h for h in logger.handlers if isinstance(h, FSSpecFileHandler)]
    for handler in file_handlers:
        handler.close()
        logger.removeHandler(handler)

    # Create new file handler
    if log_file is not None:
        # Create custom formatters
        file_formatter = CustomFormatter(
            "%(asctime)s | %(levelname)s | %(group)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = FSSpecFileHandler(log_file)
        file_handler.setLevel(level if level is not None else logger.level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)


# Create a single instance of the logger
octo_logger = setup_logger()


def get_logger():
    """Get the singleton logger instance."""
    return octo_logger
