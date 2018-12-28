import functools


def maybe_mkdir(path, warn=False):
    """ Create directory when it didn't exist."""
    import os
    import warnings

    try:
        os.makedirs(path)
    except FileExistsError as e:
        if warn:
            warnings.warn(e)


def check_same_length(f):
    """ A decorator that checks all the arguments to be the same length"""
    f_name = f.__name__
    @functools.wraps(f)
    def wrapped(*args):
        assert all(hasattr(i, '__len__') for i in args), 'Not all the arguments has attribute __len__'

        lens = [len(i) for i in args]
        if len(set(lens)) > 1:
            raise ValueError('{} expecting all the arguments to have same length.'.format(f_name))
        return f(*args)
    return wrapped




