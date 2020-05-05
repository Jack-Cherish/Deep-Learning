import logging

def get_logger(LEVEL, log_file = None):
    head = '[%(asctime)-15s] [%(levelname)s] %(message)s'
    if LEVEL == 'info':
        logging.basicConfig(level=logging.INFO, format=head)
    elif LEVEL == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=head)
    logger = logging.getLogger()
    if log_file != None:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
    return logger

logger = get_logger('info')

logger.info('Jack Cui')
logger.info('https://cuijiahua.com')
logger.info('https://mp.weixin.qq.com/s/OCWwRVDFNslIuKyiCVUoTA')