import logging
import os


def get_logger(name, log_dir):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, "{}.info.log".format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(
        info_name, when="D", encoding="utf-8"
    )
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = "#----------Config info----------#"
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == "_":
            continue
        else:
            log_info = f"{k}: {v},"
            logger.info(log_info)
