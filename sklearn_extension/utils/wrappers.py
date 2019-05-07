import re
import pandas as pd
import numpy as np
import functools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


__all__ = ['return_frame', 'ConditionalWrapper']


def _pass_index_and_columns(f):
    @functools.wraps(f)
    def wrapped_fit_method(*args, **kwargs):
        # find X
        if 'X' in kwargs:
            X = kwargs['X']
        else:
            X = args[1]

        idx, cols = X.index, X.columns
        res = f(*args, **kwargs)
        if not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res, index=idx, columns=cols)
        return res

    return wrapped_fit_method


def return_frame(cls):
    """ A class decorator for Scikit-Learn transformers
        that make sure the transform() method returns a DataFrame with same index and columns.
    """
    if not issubclass(cls, BaseEstimator) and hasattr(cls, 'transform'):
        raise ValueError('Not a Scikit-Learn transformer.')
    wrapped_transform = _pass_index_and_columns(getattr(cls, 'transform'))
    setattr(cls, 'transform', wrapped_transform)
    return cls


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

        valid_index = self.find_valid_index(X[self.cols])
        y = y[valid_index] if y is not None else y
        self.estimator.fit(X.loc[valid_index, self.cols], y)
        return self

    def transform(self, X, y=None):
        x = X.copy()

        valid_index = self.find_valid_index(X[self.cols])
        if hasattr(self.estimator, 'transform'):
            res = self.estimator.transform(x.loc[valid_index, self.cols])
        elif hasattr(self.estimator, 'fit_transform'):
            # for estimators like TSNE, ISOMAP
            res = self.estimator.fit_transform(x.loc[valid_index, self.cols])
        else:
            raise AttributeError('Estimator does not have transform or fit_transform method.')

        if hasattr(self.estimator, 'n_components'):
            # dimension reduction
            col_name = [self.col_name + '_' + str(i + 1) for i in range(self.estimator.n_components)]
            for c in col_name:
                x[c] = np.nan

            x.loc[valid_index, col_name] = res
            return x.drop(self.cols, axis=1) if self.drop else x
        else:
            x.loc[valid_index, self.cols] = res
            return x

    def __repr__(self):
        return 'Conditional<' + \
               re.sub(r'\(.*\)', '(cols={})'.format(repr(self.cols)), repr(self.estimator), flags=re.DOTALL) + \
               '>'

    def __getattr__(self, item):
        return getattr(self.estimator, item)

