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


def print_source_code(obj):
    """ Print the source code of an object."""
    import inspect
    print(inspect.getsource(obj))


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


def ngram(iterable, n=2):
    """ Generating n-gram from iterable."""
    length = len(iterable)
    return [iterable[i:i+n] for i in range(length-n+1)]


def na_default(default_value):
    """ A decorator that checks the first argument, if it's nan return the default value."""
    def is_nan(x):
        """ A copy of sklearn.utils.is_scalar_nan"""
        import numbers
        import numpy as np
        return bool(isinstance(x, (numbers.Real, np.floating)) and np.isnan(x))

    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            if is_nan(args[0]):
                return default_value
            else:
                return f(*args, **kwargs)
        return wrapped
    return decorator
