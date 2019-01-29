import contextlib
import pandas as pd


@contextlib.contextmanager
def capture_output():
    """ Capture stdout and stderr as string.
    ex.
    with capture_output() as stream:
        print('a')
    print(stream)   # [[], ['a']]
    """
    import sys
    from io import StringIO

    olderr, oldout = sys.stderr, sys.stdout
    try:
        out = [StringIO(), StringIO()]
        sys.stderr, sys.stdout = out
        yield out
    finally:
        sys.stderr, sys.stdout = olderr, oldout
        out[0] = out[0].getvalue().splitlines()
        out[1] = out[1].getvalue().splitlines()


def read_csv(path, **kwargs):
    """ Read multiple csv files and stack them rowwise."""
    from glob import glob
    files = [pd.read_csv(f, **kwargs) for f in glob(path)]
    return pd.concat(files, axis=0)
