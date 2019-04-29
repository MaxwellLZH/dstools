import pandas as pd
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import is_scalar_nan
from ..utils import searchsorted, assign_group, map_series


class Binning(BaseEstimator, TransformerMixin):
    """ Base class for all Binning functionalities,
        Subclasses should overwrite the _fit and _transform method for their own purporses.
    """
    def __init__(self,
                 cols=None,
                 bins=None,
                 encode: bool = True,
                 fill: int = -1):
        """ bins is a dictionary mapping column names to its transformation rule. The transformation rule
        can either be a list of cutoff points or a dictionary of value mapping.
        If a column is specified in the `bins` argument, it will not be fitted during the `fit` method.
        :param cols: A list of columns to perform binning, if set to None, perform binning on all columns.
        :param encode: If set to False, the result of transform will be right cutoff point of the interval
        :param fill: Used to fill in missing value.
        """
        self.cols = cols
        # self.set_bins is used to store all the user_specified cutoffs
        self.set_bins = bins or dict()
        # self.bins is used to track all the cutoff points
        self.bins = None
        self.encode = encode
        self.fill = fill

    def _fit(self, X: pd.Series, y, **fit_parmas):
        """ Fit a single feature and return the cutoff points or None,
            If None is returned, that column will not be binned.
        """
        raise NotImplementedError

    def _transform(self, X: pd.Series, y=None):
        """ Transform a single feature which has been fitted, aka the _fit method
            returns cutoff points rather than None
        """
        col_name = X.name
        rule = self.bins[col_name]
        binned = assign_group(X, rule)

        if self.encode:
            return searchsorted(rule, binned, self.fill)
        else:
            return binned

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """
        :param X: Pandas DataFrame with shape (n_sample, n_feature)
        :param y: a label column with shape (n_sample, )
        """
        self.cols = self.cols or X.columns.tolist()
        self.bins = dict()

        for col in self.cols:
            # use the user specified cutoff point
            if col in self.set_bins:
                if isinstance(self.set_bins[col], list):
                    self.bins[col] = sorted(self.set_bins[col])
                else:
                    self.bins[col] = self.set_bins[col]
                continue

            cutoff = self._fit(X[col], y)
            if cutoff is not None:
                # save the sorted cutoff points
                self.bins[col] = sorted(cutoff)
            else:
                # save a mapping from value to encoding value (starting from 1)
                self.bins[col] = {v: (k+1) for k, v in enumerate(X[col].unique()) \
                                     if not is_scalar_nan(v)}
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.bins is None:
            raise NotFittedError('This {} is not fitted. Call the fit method first.'.format(self.__class__.__name__))
        x = X.copy()
        for col in self.cols:
            if col not in self.bins:
                raise ValueError('{} was not seen during the fit process'.format(col))
            else:
                if isinstance(self.bins[col], list):
                    # rule is the cutoff points
                    x[col] = self._transform(x[col])
                else:
                    # rule is the mapping, set any unseen categories to 0
                    mapping = self.bins[col]
                    x[col] = map_series(x[col], mapping, 0, self.fill)
        return x

    def get_interval_mapping(self, col_name: str):
        """ Get the mapping from encoded value to its corresponding group. """
        if self.bins is None:
            raise NotFittedError('This {} is not fitted. Call the fit method first.'.format(self.__class__.__name__))

        if col_name not in self.bins:
            raise ValueError('Column {} was not seen during the fit process'.format(col_name))

        rule = self.bins[col_name]
        if isinstance(rule, list):
            length = len(rule)
            interval = enumerate([rule[i:i + 2] for i in range(length - 1)])
            # the first interval is close on both ends
            interval = ['[' + ', '.join(map(str, j)) + ']' if i == 0 else
                        '(' + ', '.join(map(str, j)) + ']'
                        for i, j in interval]
            interval = ['(-inf, {})'.format(min(rule))] + interval + ['({}, inf)'.format(max(rule))]
            mapping = dict(enumerate(interval))
            mapping[self.fill] = 'MISSING'
            return mapping
        else:
            mapping = defaultdict(list)
            for k, v in rule.items():
                mapping[v].append(k)

            mapping = {k: '[' + ', '.join(map(str, v)) + ']' for k, v in mapping.items()}
            mapping[self.fill] = 'MISSING'
            mapping[0] = 'UNSEEN'
            return mapping