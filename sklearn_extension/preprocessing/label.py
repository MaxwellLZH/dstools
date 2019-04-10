from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import check_is_fitted
import pandas as pd

from ..feature_selection import woe
from ..utils import encode_with_nearest_key


class WoeEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):
        """
        :param cols: A list of column names to apply transformations, default for all the columns
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X: pd.DataFrame, y):
        # missing value can not be handled by WoeEncoder since np.nan will fail the equality check
        assert_all_finite(X)
        # store a mapping from feature value to woe value
        self.mapping_ = dict()
        self.inverse_mapping_ = dict()
        self.cols = self.cols or X.columns.tolist()

        for col in self.cols:
            woe_value = woe(X[col], y)
            self.mapping_[col] = woe_value
            self.inverse_mapping_[col] = pd.Series(woe_value.index, woe_value.values)
        return self

    def transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self, ['mapping_'])
        x = X.copy()
        for col in set(self.cols) & set(X.columns):
            if col not in self.mapping_:
                raise ValueError('Column {} not seen during the fit() process.'.format(col))
            # TODO: Better way to deal with values that didn't appear in the fit() process
            # if value didn't appear in the fit() process find the nearest value for it
            # x[col] = x[col].map(lambda x: encode_with_nearest_key(self.mapping_[col], x))
            x[col] = x[col].map(self.mapping_[col], x).fillna(0)
        return x

    def inverse_transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self, ['mapping_'])
        x = X.copy()
        for col in self.cols:
            if col not in self.mapping_:
                raise ValueError('Column {} not seen during the fit() process.'.format(col))
            x[col] = x[col].map(self.inverse_mapping_[col])
        return x


if __name__ == '__main__':
    import random
    X = pd.DataFrame({'a': [random.randint(1, 3) for _ in range(10000)],
                      'b': [random.randint(1, 3) for _ in range(10000)],
                      'c': [random.randint(1, 3) for _ in range(10000)]})
    y = [int(random.random() > 0.5) for _ in range(10000)]

    WE = WoeEncoder(cols=['a', 'b'])
    encoded = WE.fit_transform(X, y)
    decoded = WE.inverse_transform(encoded)
    print(encoded)