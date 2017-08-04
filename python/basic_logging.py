import logging


def setup_basic_logging():
    """Sets up basic logging to stdout"""
    logging.basicConfig(
        format="%(asctime)-15s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
