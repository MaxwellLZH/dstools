""" Implements the equal width and equal frequency binning """
import pandas as pd
import numpy as np
import warnings
from pandas.api.types import is_numeric_dtype

from ..utils import searchsorted, wrap_with_inf, assign_group, make_series
from .base import Binning


class EqualWidthBinning(Binning):
    def __init__(self,
                 n: int,
                 bins: dict = None,
                 encode: bool = True,
                 fill: int = -1):
        """
        :param n: Number of bins to split into
        :param bins: A dictionary mapping column name to cutoff points
        :param encode: If set to False, the result of transform will be right cutoff point of the interval
            If the input has missing values, it will be put under a seperate group with the largest bin value
        :param fill: Used to fill in missing value.
        """
        super().__init__(bins, encode, fill)
        self.n = n

    def _fit(self, X: pd.Series, y=None, **fit_parmas):
        """ Fit a single feature and return the cutoff points"""
        if not is_numeric_dtype(X):
            return None

        def find_nearest_element(series, elem):
            min_idx = (series - elem).abs().values.argmin()
            return series.iloc[min_idx]

        X_ = X[X.notnull()]
        v_min, v_max = X_.min(), X_.max()
        bins = [find_nearest_element(X_, elem) for elem in np.linspace(v_min, v_max, self.n+1)]
        return bins


class EqualFrequencyBinning(Binning):

    def __init__(self,
                 n: int,
                 bins: dict = None,
                 encode: bool = True,
                 fill: int = -1):
        """
        :param q: Number of equal width intervals to split into
        :param bins: A series of cutoff points, if provided, n will be ignored
        :param encode: If set to False, the result of transform will be right cutoff point of the interval
        :param fill: Used to fill in missing value.
        """
        super().__init__(bins, encode, fill)
        self.n = n

    def _fit(self, X: pd.Series, y=None, **fit_parmas):
        """ Fit a single feature and return the cutoff points"""
        if not is_numeric_dtype(X):
            return None

        quantiles = np.linspace(0, len(X[X.notnull()]) - 1, self.n+1, dtype=int)
        cutoff = X.sort_values().reset_index(drop=True)[quantiles]
        # there might be duplicated cutoff points
        return set(cutoff)


def equal_width_binning(X: pd.Series, n: int, encode: bool = True, fill: int = -1):
    """ Shortcut for equal width binning on a Pandas.Series, returns
        the encoded series and the cutoff points
    """
    s_name = X.name or 0
    EWB = EqualWidthBinning(n, encode=encode, fill=fill)
    binned = EWB.fit_transform(X.to_frame())
    return binned[s_name], EWB.bins[s_name]


def equal_frequency_binning(X: pd.Series, n: int, encode: bool = True, fill: int = -1):
    """ Shortcut for equal frequency binning on a Pandas.Series, returns
        the encoded series and the cutoff points
    """
    s_name = X.name or 0
    EFB = EqualFrequencyBinning(n, encode=encode, fill=fill)
    binned = EFB.fit_transform(X.to_frame())
    return binned[s_name], EFB.bins[s_name]


if __name__ == '__main__':
    # s = pd.Series(list(range(20)) + [np.nan] * 4)
    # EWB = EqualWidthBinning(n=5, encode=True)
    # EFB = EqualFrequencyBinning(n=5, encode=False)
    # print(EFB.fit_transform(s))
    # print(EFB.bins)
    # print(EFB.transform(s))
    pass

