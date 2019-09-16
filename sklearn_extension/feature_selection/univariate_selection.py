import pandas as pd
import numpy as np

from sklearn.feature_selection.base import SelectorMixin
from sklearn.base import BaseEstimator

from ..utils import make_series
from ..linear_model import StepwiseLogisticRegression


def woe(X, y, conditional=False, na_values=None) -> pd.Series:
    """ Return a series mapping feature value to its woe stats
    :param conditional: If set to True, the part where X is missing will be excluded from the calculation
    :param na_values: Values that should be treated as NaN
    """
    X, y = make_series(X), make_series(y)
    if conditional:
        if na_values is None:
            mask = pd.notnull(X)
        elif isinstance(na_values, (list, tuple)):
            from pandas.core.algorithms import isin
            mask = ~isin(X, na_values)
        else:
            mask = X != na_values
        X, y = X[mask], y[mask]

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


class SelectStepwise(BaseEstimator, SelectorMixin):
    """ Feature selection based on stepwise logistic regression """

    def __init__(self, 
                k=None,
                alpha_enter=0.15,
                alpha_exit=0.15,
                criteria='p-value',
                mode='bidirectional',
                method='bfgs'):
        self.StepwiseLR = StepwiseLogisticRegression(alpha_enter=alpha_enter,
                                                alpha_exit=alpha_exit,
                                                criteria=criteria,
                                                max_feature=k,
                                                mode=mode,
                                                method=method,
                                                refit=False)
    def fit(self, X, y):
        self.StepwiseLR.fit(X, y)
        self.model_cols = self.StepwiseLR.model_cols
        self.cols = X.columns.tolist()
        return self

    def _get_support_mask(self):
        return np.array([c in self.model_cols for c in self.cols], dtype=bool)
