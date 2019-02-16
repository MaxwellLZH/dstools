import contextlib
import pandas as pd
import functools


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



def save_to(path, n_obj=1, mode='auto'):
    def decorator(f):
        import os
        
        f_name = f.__name__
        dir_name = os.path.dirname(path)
        file_name = os.path.basename(path).split('.')[0]


        def save_file(obj, out_path, mode, suffix=''):
            _save_mode = {
                pd.DataFrame: 'csv',
                str: 'txt',
                list: 'txt',
                tuple: 'txt',
                dict: 'txt',
            }

            if mode == 'auto':
                mode = _save_mode.get(type(obj), 'pkl')

            out_path = out_path + suffix + '.' + mode
            if os.path.exists(out_path):
                raise ValueError('File {} has already exists'.format(outpath))

            if mode == 'csv':
                if isinstance(obj, pd.DataFrame):
                    obj.to_csv(out_path)
                else:
                    raise ValueError('Only support saving Pandas DataFrame as csv file.')


        def save(*args, **kwargs):
            result = f(*args, **kwargs)

            out_path = os.path.join(dir_name, path_name)
            if n_obj == 1:
                save_file(result, out_path, mode)
            else:
                # just in case an generator is returned
                result = list(result)
                n_result = len(result)
                if n_result != n_obj:
                    raise ValueError('The number of outputs {} does not match with expected '
                            'number of output {}'.format(n_result, n_obj))

                for i, obj in enumerate(result):
                    suffix = '_' + str(i)
                    save_file(obj, out_path, mode, suffix)

            return result

        return save
    return decorator








