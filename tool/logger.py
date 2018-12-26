import logging.config

from tool.logging_conf import MY_LOGGING_CONF

logging.config.dictConfig(MY_LOGGING_CONF)
logger = logging.getLogger("other")
