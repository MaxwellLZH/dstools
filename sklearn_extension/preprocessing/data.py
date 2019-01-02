from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, assert_all_finite
import pandas as pd
import warnings
from types import FunctionType, LambdaType

from sklearn.utils import check_array
from sklearn.utils import column_or_1d


def _encode_python(values, uniques=None, encode=False, unseen='warn'):
    """ Modified from the Scikit version except handles unseen values"""
    if uniques is None:
        uniques = sorted(set(values))
        uniques = np.array(uniques, dtype=values.dtype)
    if encode:
        table = {val: i for i, val in enumerate(uniques)}
        try:
            encoded = np.array([table[v] for v in values])
        except KeyError as e:
            msg = "y contains previously unseen labels: %s" % str(e)
            if unseen in ('silent', 'warn'):
                UNSEEN = len(uniques)
                encoded = np.array([table.get(v, UNSEEN) for v in values])
                if unseen == 'warn':
                    warnings.warn(msg)
            elif unseen == 'raise':
                raise ValueError("y contains previously unseen labels: %s"
                                 % str(e))
            else:
                raise ValueError('The supported options for `unseen` are: {}'
                                 .format(['silent', 'warn', 'raise']))
        return uniques, encoded
    else:
        return uniques


class NormDistOutlierRemover(BaseEstimator, TransformerMixin):
    """ Removing outliers assuming data is independent and followes normal distribution
        Each series will be within the range of [mu - n_sigma * std, mu + n_sigma * std]
    """
    def __init__(self, n_sigma=3, cols=None, error='warn'):
        """
        :param cols: A list of column names to apply transformations, default for all the columns
        :param error: Specify the action when the DataFrame passed to transform doesn't have all the columns,
            supported actions are ['raise', 'ignore', 'warn']
        """
        self.n_sigma = n_sigma
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.error = error

    def fit(self, X: pd.DataFrame, y=None):
        X = X[self.cols] if self.cols else X
        self.cols = self.cols or X.columns

        self.mean_ = X.mean()
        self.std_ = X.std()
        return self

    def transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self, ['mean_', 'std_'])
        x = X.copy()

        # copy instance variable to local variable
        _mean, _std, n_sigma = self.mean_, self.std_, self.n_sigma
        for col in self.cols:
            if col not in x:
                msg = 'Column {} is not found in the DataFrame'.format(col)
                if self.error == 'raise':
                    raise ValueError(msg)
                if self.error == 'warn':
                    warnings.warn(msg)

            lower_bound = _mean[col] - n_sigma * _std[col]
            upper_bound = _mean[col] + n_sigma * _std[col]
            x.loc[x[col] > upper_bound, col] = upper_bound
            x.loc[x[col] < lower_bound, col] = lower_bound
        return x


class IQROutlierRemover(BaseEstimator, TransformerMixin):
    """ Removing outlier based on IQR,
        Each series will be in the range of [Q1 - multiplier * IQR , Q3 + multiplier]
    """
    def __init__(self, multiplier=1.5, cols=None, interpolation='nearest', error='warn'):
        """
        :param interpolation: interpolation used in calculating the quantile
        :param cols: A list of column names to apply transformations, default for all the columns
        :param error: Specify the action when the DataFrame passed to transform doesn't have all the columns,
            supported actions are ['raise', 'ignore', 'warn']
        """
        self.multiplier =multiplier
        self.interpolation = interpolation
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.error = error

    def fit(self, X: pd.DataFrame, y=None):
        X = X[self.cols] if self.cols else X
        self.cols = self.cols or X.columns

        q1 = X.quantile(0.25, interpolation=self.interpolation)
        q3 = X.quantile(0.75, interpolation=self.interpolation)
        self.q2 = X.quantile(0.5, interpolation=self.interpolation)
        self.iqr = q3 - q1
        return self

    def transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self, 'iqr')
        q2, iqr, m = self.q2, self.iqr, self.multiplier
        x = X.copy()
        for col in self.cols:
            if col not in x:
                msg = 'Column {} is not found in the DataFrame'.format(col)
                if self.error == 'raise':
                    raise ValueError(msg)
                if self.error == 'warn':
                    warnings.warn(msg)

            lower_bound = q2[col] - m * iqr[col]
            upper_bound = q2[col] + m * iqr[col]
            x.loc[x[col] > upper_bound, col] = upper_bound
            x.loc[x[col] < lower_bound, col] = lower_bound
        return x


class QuantileOutlierRemover(BaseEstimator, TransformerMixin):
    """ Removing outlier when it's above or below a certain quantile,
    """
    def __init__(self, quantile=0.95, upper=True, cols=None, interpolation='nearest', error='warn'):
        """
        :param quantile: The quantile above or below which should be considered outlier
        :param upper: The upper or lower end to be considered as outlier
        :param interpolation: interpolation used in calculating the quantile
        :param cols: A list of column names to apply transformations, default for all the columns
        :param error: Specify the action when the DataFrame passed to transform doesn't have all the columns,
            supported actions are ['raise', 'ignore', 'warn']
        """
        self.quantile = quantile
        self.upper = upper
        self.interpolation = interpolation
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.error = error

    def fit(self, X: pd.DataFrame, y=None):
        X = X[self.cols] if self.cols else X
        self.cols = self.cols or X.columns
        self.threshold = X.quantile(self.quantile, interpolation=self.interpolation)
        return self

    def transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self, 'threshold')
        threshold = self.threshold
        x = X.copy()
        for col in self.cols:
            if col not in x:
                msg = 'Column {} is not found in the DataFrame'.format(col)
                if self.error == 'raise':
                    raise ValueError(msg)
                if self.error == 'warn':
                    warnings.warn(msg)

            th = threshold[col]
            if self.upper:
                x.loc[x[col] > th, col] = th
            else:
                x.loc[x[col] < th, col] = th
        return x


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """ Similar Scikit-Learn OrdinalEncoder but allows for arbitrary ordering in the columns,
        also handles unseen values during transform() process.
    """
    def __init__(self, cols=None, fill=True, error='warn', unseen='warn'):
        """
        :param cols: A list of column names to apply transformations, default for all the columns
        :param fill: Whether to fill missing value before encoding
        :param error: Specify the action when the DataFrame passed to transform doesn't have all the columns,
            supported actions are ['raise', 'ignore', 'warn']
        :param unseen: Specify the action when encountered with unseen values,
            supported actions are ['raise, 'silent', 'warn']
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.fill = fill
        self.error = error
        self.unseen = unseen

    def fit(self, X: pd.DataFrame, y=None):
        X = X[self.cols] if self.cols else X
        if not self.fill:
            assert_all_finite(X, allow_nan=False)
        else:
            X = X.copy().fillna('_MISSING')
        self.categories_ = dict()

        for col in X:
            cutoff = _encode_python(X[col].astype(str))
            self.categories_[col] = cutoff
        return self

    def transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self, 'categories_')
        x = X.copy()

        for col in self.cols:
            if col not in x:
                msg = 'Column {} is not found in the DataFrame'.format(col)
                if self.error == 'raise':
                    raise ValueError(msg)
                if self.error == 'warn':
                    warnings.warn(msg)

            if not self.fill:
                assert_all_finite(x[col], allow_nan=False)
            else:
                x[col] = x[col].fillna('_MISSING').astype(str)
            cutoff = self.categories_[col]
            _, x[col] = _encode_python(x[col], uniques=cutoff, encode=True, unseen=self.unseen)
        return x


if __name__ == '__main__':
    import random
    import numpy as np

    X = pd.DataFrame({'a': list(range(1000)),
                      'b': [random.randint(5, 10) for _ in range(1000)],
                      'c': [random.randint(1, 100) for _ in range(1000)]})
    Encoder = QuantileOutlierRemover(cols=['a', 'b'])
    print(Encoder.fit_transform(X))
