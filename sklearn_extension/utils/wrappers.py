import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class ConditionalWrapper(BaseEstimator, TransformerMixin):
    """ A conditional wrapper that makes a Scikit-Learn transformer only works on part of the data
        where X is not missing.
    """
    def __init__(self, estimator, cols=None, na_values=None, col_name=None, drop=True):
        """
        :param na_values: Values that should be treated as NaN
        :param col_name: If the estimator reduces the dimension, then the new columns
            will be assigned {col_name}_1, {col_name}_2, {col_name}_3 ......
        :param drop: If the estimator reduces the dimension, determines whether the original
            columns should be dropped
        """
        self.estimator = estimator
        self.cols = cols
        self.na_values = na_values
        self.valid_index = None
        self.col_name = col_name
        self.drop = drop

    def find_valid_index(self, X: pd.DataFrame):
        if self.na_values is None:
            valid_index = pd.notnull(X).all(axis=1)
        elif isinstance(self.na_values, (list, tuple)):
            from pandas.core.algorithms import isin
            valid_index = X.apply(lambda x: ~isin(x, self.na_values)).all(axis=1)
        else:
            valid_index = X.apply(lambda x: x != self.na_values).all(axis=1)
        return valid_index

    def fit(self, X, y=None, **fit_params):
        if self.cols is None:
            self.cols = X.columns.tolist()
        else:
            self.cols = [c for c in self.cols if c in X.columns]

        self.valid_index = valid_index = self.find_valid_index(X[self.cols])
        y = y[valid_index] if y is not None else y
        self.estimator.fit(X.loc[valid_index, self.cols], y)
        return self

    def transform(self, X, y=None):
        if self.valid_index is None:
            raise NotFittedError('The ConditionalWrapper is not fitted yet.')
        x = X.copy()
        res = self.estimator.transform(x.loc[self.valid_index, self.cols])

        if hasattr(self.estimator, 'n_components'):
            # dimension reduction
            col_name = [self.col_name + '_' + str(i + 1) for i in range(self.estimator.n_components)]
            for c in col_name:
                x[c] = np.nan

            x.loc[self.valid_index, col_name] = res
            return x.drop(self.cols, axis=1) if self.drop else x
        else:
            x.loc[self.valid_index, self.cols] = res
            return x

    def __repr__(self):
        return 'Conditional<' + \
               re.sub(r'\(.*\)', '(cols={})'.format(repr(self.cols)), repr(self.estimator), flags=re.DOTALL) + \
               '>'

    def __getattr__(self, item):
        return getattr(self.estimator, item)