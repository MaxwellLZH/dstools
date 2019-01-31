from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import column_or_1d
import pandas as pd
import numpy as np


def force_zero_one(y):
    """" Convert two-class labels to 0 and 1
    ex. [-1, 1, -1, -1, 1] => [0, 1, 0, 0, 1]
    """
    y = column_or_1d(y, warn=True)
    if len(set(y)) != 2:
        raise ValueError('The label should have exactly two categories')
    return LabelBinarizer().fit_transform(y).ravel()


def make_series(i, reset_index=True) -> pd.Series:
    """ Convert an iterable into a Pandas Series"""
    if isinstance(i, pd.DataFrame):
        series = i.iloc[:, 0]
    else:
        series = pd.Series(i)
    # make index starts from 0
    if reset_index:
        series = series.reset_index(drop=True)
    return series


def _searchsorted(a, v):
    """ Same as np.searchsorted(a, v, side='left') but faster for our purpose."""
    for i, c in enumerate(a):
        if c >= v:
            return i
    return len(a)


def searchsorted(a, v, fill=-1):
    """ Encode values in v with ascending cutoff points in a. Similar to numpy.searchsorted
        Left open right close except for the leftmost interval, which is close at both ends.
    """
    encoded = list()
    for value in v:
        if np.isnan(value):
            encoded.append(fill)
        elif value == min(a):
            # the leftmost interval close at both ends
            encoded.append(1)
        else:
            encoded.append(_searchsorted(a, value))
    return encoded


def assign_group(x, bins):
    """ Assign the right cutoff value for each value in x given the bins
    """
    # add infinite at the end
    extended_cutoff = list(bins) + [np.inf]
    return [v if np.isnan(v) else extended_cutoff[_searchsorted(extended_cutoff, v)] \
                                                 for v in x]


def wrap_with_inf(bins):
    """ Given a series of cutoff points, add positive and negative infinity
        at both ends of the cutoff points
     """
    return np.unique(list(bins) + [np.inf, -np.inf])

