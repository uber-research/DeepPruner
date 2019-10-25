import logging

def setup_logging(filename):

    log_format = '%(filename)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(logging.Formatter(fmt=log_format))
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
    