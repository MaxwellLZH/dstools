import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class Binning(BaseEstimator, TransformerMixin):
    """ Base class for all Binning functionalities,
        Subclasses should overwrite the _fit and _transform method for their own purporses.
    """
    def __init__(self, bins=None):
        """ bins is a dictionary mapping column names to its cutoff points"""
        self.bins = bins or dict()

    def _fit(self, X, y, **fit_parmas):
        """ Fit a single feature and return the cutoff points or None,
            If None is returned, that column will not be binned.
        """
        raise NotImplementedError

    def _transform(self, X, y=None):
        """ Transform a single feature"""
        raise NotImplementedError

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
