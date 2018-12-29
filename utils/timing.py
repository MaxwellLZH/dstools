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


def get_stats(statement):
    """ Return a pstats.Stats object from a statement. """
    import os
    import uuid
    import cProfile as profiler
    import pstats

    tmp_file = './{}.statas'.format(uuid.uuid1())
    profiler.run(statement, filename=tmp_file)
    stats = pstats.Stats(tmp_file)
    os.remove(tmp_file)
    stats.strip_dirs()
    return stats


def print_stats(statement, *keys):
    """ Print out the profiling detail from the statement sorted by *keys"""
    keys = keys or ['cumulative']
    stats = get_stats(statement)
    stats.sort_stats(*keys)
    stats.print_stats()