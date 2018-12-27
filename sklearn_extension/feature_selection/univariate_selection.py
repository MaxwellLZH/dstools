import numpy as np
from sklearn_extension.utils import make_series
import pandas as pd


def woe(X, y) -> pd.Series:
    """ Return a series mapping feature value to its woe stats"""
    X, y = make_series(X), make_series(y)
    total_pos = y.sum()
    total_neg = len(y) - total_pos
    pct_pos = (y.groupby(X).sum() + 1) / (total_pos + 1)
    pct_neg = (y.groupby(X).count() - y.groupby(X).sum() + 1) / (total_neg + 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        woe_value = np.log(pct_pos / pct_neg)
    return woe_value


def _iv(X, y) -> float:
    """ Calulate the iv value for a single feature"""
    X, y = make_series(X), make_series(y)
    total_pos = y.sum()
    total_neg = len(y) - total_pos
    pct_pos = (y.groupby(X).sum() + 1) / (total_pos + 1)
    pct_neg = (y.groupby(X).count() - y.groupby(X).sum() + 1) / (total_neg + 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        woe = np.log(pct_pos / pct_neg)
        iv = (pct_pos - pct_neg) * woe
    return iv.sum()


def iv(X, y) -> np.array:
    """ Compute the iv stats for each feature, return a list of woe value."""
    return np.apply_along_axis(lambda x: _iv(x, y), 0, X)


