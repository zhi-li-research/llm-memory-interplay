import logging
import pprint

LOGGER_NAME = "causal_tracer"

logger = logging.getLogger(LOGGER_NAME)

def log_or_print(msg, verbose, level=logging.INFO):
    if verbose:
        pprint.pp(msg)
    else:
        logger.log(level, msg)
