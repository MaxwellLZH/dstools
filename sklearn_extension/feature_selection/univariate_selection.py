import pandas as pd
import numpy as np

from ..utils import make_series


def woe(X, y) -> pd.Series:
    """ Return a series mapping feature value to its woe stats"""
    y = make_series(y)
    total_pos = y.sum()
    total_neg = len(y) - total_pos
    grouped = y.groupby(X)
    grouped_sum = grouped.sum()
    grouped_count = grouped.count()

    pct_pos = (grouped_sum + 1) / (total_pos + 1)
    pct_neg = (grouped_count - grouped_sum + 1) / (total_neg + 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        woe_value = np.log(pct_pos / pct_neg)
    return woe_value


def _iv(X, y) -> float:
    """ Calulate the iv value for a single feature"""
    y = make_series(y)
    total_pos = y.sum()
    total_neg = len(y) - total_pos
    grouped = y.groupby(X)
    grouped_sum = grouped.sum()
    grouped_count = grouped.count()

    pct_pos = (grouped_sum + 1) / (total_pos + 1)
    pct_neg = (grouped_count - grouped_sum + 1) / (total_neg + 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        woe = np.log(pct_pos / pct_neg)
        iv = (pct_pos - pct_neg) * woe
    return iv.sum()


def iv(X, y) -> np.array:
    """ Compute the iv stats for each feature, return a list of woe value."""
    return np.apply_along_axis(lambda x: _iv(x, y), 0, X)