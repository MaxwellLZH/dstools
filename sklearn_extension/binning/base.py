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
        # list of columns that wasn't binned
        self.skip_cols = list()
        self.skip_cols_encoding = dict()

    def _fit(self, X: pd.Series, y, **fit_parmas):
        """ Fit a single feature and return the cutoff points or None,
            If None is returned, that column will not be binned.
        """
        raise NotImplementedError

    def _transform(self, X: pd.Series, y=None):
        """ Transform a single feature"""
        col_name = X.name
        binned = assign_group(X, self.bins[col_name])

        if self.encode:
            return searchsorted(self.bins[col_name], binned, self.fill)
        else:
            return binned

    def _encode(self, X: pd.Series):
        """" Encode an un-binned column """
        col_name = X.name
        mapping = self.skip_cols_encoding[col_name]
        encoded = X.map(mapping).fillna(self.fill)
        # assign 0 for unseen elements
        encoded[~X.isin(mapping)] = 0
        return encoded

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """
        :param X: Pandas DataFrame with shape (n_sample, n_feature)
        :param y: a label column with shape (n_sample, )
        """
        self.bins = dict()
        for col in X.columns:
            cutoff = self._fit(X[col], y)
            if cutoff is not None:
                # sort the cutoff points
                self.bins[col] = sorted(cutoff)
            else:
                # save the col in `self.skip_cols` and create an
                # mapping dictionary under `self.skip_cols_encoding`
                self.skip_cols.append(col)
                column = X[col]
                column = column[column.notnull()].unique()
                self.skip_cols_encoding[col] = {v: k+1 for k, v in enumerate(column)}
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.bins is None:
            raise NotFittedError('This {} is not fitted. Call the fit method first.'.format(self.__class__.__name__))
        x = X.copy()
        for col in x.columns:
            if col not in self.bins and col not in self.skip_cols:
                raise ValueError('{} was not seen during the fit process'.format(col))
            elif col in self.skip_cols:
                x[col] = self._encode(x[col])
            else:
                x[col] = self._transform(x[col])
        return x

    def get_interval_mapping(self, col_name: str):
        """ Get the mapping from encoded value to its corresponding group. """
        if self.bins is None:
            raise NotFittedError('This {} is not fitted. Call the fit method first.'.format(self.__class__.__name__))

        if col_name not in self.bins and col_name not in self.skip_cols:
            raise ValueError('Column {} was not seen during the fit process'.format(col_name))
        elif col_name in self.bins:
            # binned columns
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
        else:
            # skipped columns
            mapping = {v: k for k, v in self.skip_cols_encoding[col_name].items()}
            mapping[self.fill] = 'MISSING'
            mapping[0] = 'UNSEEN'
            return mapping
