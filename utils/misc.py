def maybe_mkdir(path, warn=False):
    import os
    import warnings

    try:
        os.makedirs(path)
    except FileExistsError as e:
        if warn:
            warnings.warn(e)