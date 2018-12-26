# -*- coding:utf-8 -*-

import os
from datetime import date

PATH = "log/"


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


assure_path_exists(PATH)

MY_LOGGING_CONF = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "company": {
            "format": "[%(asctime)s]- %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"
        },
        "other": {
            "format": "[%(asctime)s] [%(levelname)s] [%(threadName)s:%(thread)d] [%(filename)s:%(lineno)d] [] - %(message)s"
        }
    },

    "handlers": {
        "other_console_handler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "other",
            "stream": "ext://sys.stdout"
        },
        "other_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "other",
            "filename": PATH + date.today().isoformat() + ".log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },

    },
    'loggers': {
        "": {
            "level": "INFO",
            "handlers": ["pingan_console_handler", "pingan_file_handler"]
        }
    }

}
