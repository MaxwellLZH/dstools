""" Implements the equal width and equal frequency binning """
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np
import warnings

from ..utils import searchsorted, wrap_with_inf, assign_group, make_series


class EqualWidthBinning(BaseEstimator, TransformerMixin):
    """ A wrapper for Pandas.cut, the only difference is it returns the
     encoded feature rather than a categorical series
     """
    def __init__(self, n, bins=None, encode=True, fill=-1,
                 right=True, include_lowest=False, duplicates='raise'):
        """
        :param n: Number of bins to split into
        :param bins: A series of cutoff point.
        :param encode: If set to False, the result of transform will be right cutoff point of the interval
            If the input has missing values, it will be put under a seperate group with the largest bin value
        :param fill: Used to fill in missing value.
        """
        self.n = n
        self.bins = bins
        self.encode = encode
        self.fill = fill
        self.right = right
        self.include_lowest = include_lowest
        self.duplicates = duplicates

    def fit(self, X, y=None, **fit_params):
        if self.bins is not None:
            warnings.warn('The binner has already be fitted. '
                          'Calling the fit method will refit the binner.')
        _, bins = pd.cut(X, self.n, right=self.right, retbins=True,
                              include_lowest=self.include_lowest,
                              duplicates=self.duplicates)
        self.bins = wrap_with_inf(bins)
        return self

    def transform(self, X, y=None):
        if self.bins is None:
            raise NotFittedError('This EqualWidthBinner is not fitted. Call the fit method first.')
        binned = pd.cut(X, self.bins, right=self.right,
                        include_lowest=self.include_lowest,
                        duplicates=self.duplicates)
        binned = [i if i is np.nan else i.right for i in binned]

        if self.encode:
            return searchsorted(self.bins, binned, self.fill)
        else:
            return binned


class EqualFrequencyBinning(BaseEstimator, TransformerMixin):
    """ A wrapper for Pandas.qcut"""
    def __init__(self, n, bins=None, outlier_pct=1.0, encode=True, fill=-1, duplicates='raise'):
        """
        :param q: Number of equal width intervals to split into
        :param bins: A series of cutoff points, if provided, n will be ignored
        :param outlier_pct: The quantity percentage above which all the values will be in the same group
        :param encode: If set to False, the result of transform will be right cutoff point of the interval
        :param fill: Used to fill in missing value.
        """
        self.n = n
        self.bins = bins
        self.outlier_pct = outlier_pct
        self.encode = encode
        self.fill = fill
        self.duplicates = duplicates

    def fit(self, X, y=None, **fit_params):
        X = make_series(X)
        if self.bins is not None:
            warnings.warn('The binner has already be fitted. '
                          'Calling the fit method will refit the binner.')
        # _, bins = pd.qcut(X, self.n, retbins=True, duplicates=self.duplicates)
        quantiles = np.linspace(0, len(X[X.notnull()]) * self.outlier_pct - 1, self.n, dtype=int)
        cutoff = X.sort_values().reset_index(drop=True)[quantiles]
        self.bins = cutoff.values
        return self

    def transform(self, X, y=None):
        if self.bins is None:
            raise NotFittedError('This EqualFrequencyBinner is not fitted. Call the fit method first.')
        binned = assign_group(X, self.bins)

        if self.encode:
            return searchsorted(self.bins, binned, self.fill)
        else:
            return binned


if __name__ == '__main__':
    s = pd.Series(list(range(20)) + [np.nan] * 4)
    EWB = EqualWidthBinning(n=5, encode=True)
    EFB = EqualFrequencyBinning(n=5, encode=False)
    print(EFB.fit_transform(s))
    print(EFB.bins)
    print(EFB.transform(s))

