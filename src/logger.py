import logging

### Setup logging ###
def getLogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)-8s] [%(name)-8s] [%(levelname)-1s] [%(message)s]')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger