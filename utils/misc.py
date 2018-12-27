def maybe_mkdir(path, warn=False):
    """ Create directory when it didn't exist."""
    import os
    import warnings

    try:
        os.makedirs(path)
    except FileExistsError as e:
        if warn:
            warnings.warn(e)