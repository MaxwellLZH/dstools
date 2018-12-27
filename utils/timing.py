import logging
import time
import functools


def timeit(logger=None):
    """ A decorator that times the function and logs the information."""
    if logger is None:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging

    def inner(f):
        f_name = f.__name__

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            result = f(*args, **kwargs)
            end_time = time.time()
            logger.debug('{} complete, time spent: {.1f} seconds.'. \
                         format(f_name, end_time-start_time))
            return result
        return wrapped
    return inner