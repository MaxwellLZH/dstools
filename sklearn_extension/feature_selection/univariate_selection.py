
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

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


class CorrelationRemover(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None, categorical_cols=None, threshold=0.8, method='pearson', save_corr=False):
        """
        :param cols: A list of feature names, sorted by importance from high to low
        :param categorical_cols: A list of categorical column names which will all be kept at the moment
        :param threshold: The correlation upper bound
        :param method: The method used for calculating correlation
        :param save_corr: Whether to save the correlation matrix
        """
        self.cols = cols
        self.categorical_cols = categorical_cols or list()
        self.threshold = threshold
        self.method = method
        self.save_corr = save_corr
        self.mat_corr = None

        self.drop_cols = None

    def fit(self, X, y=None, **fit_params):
        """ Return the number of dropped columns """
        self.cols = cols = self.cols or X.columns.tolist()
        _error_cols = set(cols) - set(X.columns)
        if _error_cols:
            raise ValueError('The following columns are does not exist in DataFrame X: ' +
                             repr(list(_error_cols)))

        numerical_cols = list(set(X.select_dtypes(include='number')) - set(self.categorical_cols))
        if len(numerical_cols)==0:
            return self

        mat_corr = X[numerical_cols].corr(method=self.method).abs()
        if self.save_corr:
            self.mat_corr = mat_corr
            
        self.drop_cols = list()
        for i, c_a in enumerate(cols):
            for j in range(i+1, len(cols)):
                c_b = cols[j]
                if c_b in numerical_cols and \
                    c_b not in self.drop_cols and \
                    mat_corr.loc[c_a, c_b] > self.threshold:
                        self.drop_cols.append(c_b)
        
        return self

    def transform(self, X, y=None):
        if self.drop_cols is None:
            raise NotFittedError('This CorrelationRemover is not fitted. Call the fit method first.')

        drop_cols = set(X.columns) & set(self.drop_cols)
        return X.drop(drop_cols, axis=1)



