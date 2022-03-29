import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger object with the specific name.

    Args:
        name (str): [description]

    Returns:
        logging.Logger: [description]
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create file handler for logger.
    fh = logging.FileHandler('SPOT.log')
    fh.setLevel(level=logging.DEBUG)


    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger
