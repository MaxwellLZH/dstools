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


def read_multiple_files(read_fn, path, **kwargs):
    from glob import iglob
    files = [read_fn(f, **kwargs) for f in iglob(path)]
    return pd.concat(files, axis=0)


def read_csv(path, **kwargs):
    """ Read multiple csv file and concatenate them row-wise """
    return read_multiple_files(pd.read_csv, path, **kwargs)


def read_excel(path, **kwargs):
    """ Read multiple excel file and concatenate them row-wise"""
    return read_multiple_files(pd.read_excel, path, **kwargs)


def read_sheets(path, **kwargs):
    """ Read all the sheets in an excel file and concatenate them row-wise """
    sheets = pd.read_excel(path, sheet_name=None)
    return pd.concat(sheets.values(), axis=0)