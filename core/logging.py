import logging
import sys

DEFAULT_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    format: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(logging.Formatter(format, datefmt))
            handler.setLevel(level)
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(format, datefmt))
        handler.setLevel(level)
        root.addHandler(handler)

    logging.getLogger("numba").setLevel(logging.WARNING)
