import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from ..utils import searchsorted, assign_group


class Binning(BaseEstimator, TransformerMixin):
    """ Base class for all Binning functionalities,
        Subclasses should overwrite the _fit and _transform method for their own purporses.
    """
    def __init__(self,
                 bins=None,
                 encode: bool = True,
                 fill: int = -1):
        """ bins is a dictionary mapping column names to its cutoff points
        :param encode: If set to False, the result of transform will be right cutoff point of the interval
        :param fill: Used to fill in missing value.

        """
        self.bins = bins
        self.encode = encode
        self.fill = fill

    def _fit(self, X, y, **fit_parmas):
        """ Fit a single feature and return the cutoff points or None,
            If None is returned, that column will not be binned.
        """
        raise NotImplementedError

    def _transform(self, X, y=None):
        """ Transform a single feature"""
        col_name = X.name
        binned = assign_group(X, self.bins[col_name])

        if self.encode:
            return searchsorted(self.bins[col_name], binned, self.fill)
        else:
            return binned

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """
        :param X: Pandas DataFrame with shape (n_sample, n_feature)
        :param y: a label column with shape (n_sample, )
        """
        self.bins = dict()
        for col in X.columns:
            bins = self._fit(X[col], y)
            # sort the cutoff points
            self.bins[col] = sorted(bins) if bins is not None else None
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.bins is None:
            raise NotFittedError('This {} is not fitted. Call the fit method first.'.format(self.__class__.__name__))
        x = X.copy()
        for col in x.columns:
            if self.bins.get(col, None) is not None:
                x[col] = self._transform(x[col])
        return x

    def get_interval_mapping(self, col_name: str):
        """ Get the mapping from encoded value to its corresponding group. """
        if self.bins is None:
            raise NotFittedError('This {} is not fitted. Call the fit method first.'.format(self.__class__.__name__))

        if col_name not in self.bins:
            raise ValueError('Column {} was not seen during the fit process'.format(col_name))

        cutoff = self.bins[col_name]
        length = len(cutoff)
        interval = enumerate([cutoff[i:i + 2] for i in range(length - 1)])
        # the first interval is close on both ends
        interval = ['[' + ', '.join(map(str, j)) + ']' if i == 0 else
                    '(' + ', '.join(map(str, j)) + ']'
                    for i, j in interval]
        interval = ['(-inf, {})'.format(min(cutoff))] + interval + ['({}, inf)'.format(max(cutoff))]
        mapping = dict(enumerate(interval))
        mapping[self.fill] = 'MISSING'
        return mapping
