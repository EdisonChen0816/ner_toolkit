# encoding=utf-8
import logging


def setlogger(config):
    cfg = {}
    for k in config:
        cfg[k] = config[k]
    logger = logging.getLogger(cfg['Server_Name'])
    if len(logger.handlers) > 0:
        return logger
    formatter = logging.Formatter(cfg['LogFormat'])
    logger.setLevel(10)
    # setup console setting
    if cfg['LogLevel_Console'] > 0:
        ch = logging.StreamHandler()
        ch.setLevel(cfg["LogLevel_Console"])
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    # setup file logging path
    if cfg['LogLevel_File'] > 0:
        fh = logging.FileHandler(cfg['LogFile'], encoding='utf-8', mode='a')
        fh.setLevel(cfg["LogLevel_File"])
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


if __name__ == '__main__':
    config = {}
    logger = setlogger(config)
    text = input("input print info:")
    logger.info(text)
    logger.warning("it worked")
