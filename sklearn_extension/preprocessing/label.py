from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import check_is_fitted
import pandas as pd

from ..feature_selection import woe
from ..utils import encode_with_nearest_key


class WoeEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None, conditional_cols=None, na_values=None):
        """
        :param cols: A list of column names to apply transformations, default for all the columns
        :param conditional_cols: A list of columns that needs to calculate the woe value with conditional flag
        :param na_values: Values that should be treated as NaN
        """
        self.cols = cols
        self.conditional_cols = conditional_cols
        self.na_values = na_values

    def fit(self, X: pd.DataFrame, y):
        # store a mapping from feature value to woe value
        self.mapping_ = dict()
        self.cols = self.cols or X.columns.tolist()
        self.conditional_cols = self.conditional_cols or []

        for col in self.cols:
            if col not in self.conditional_cols:
                # missing value can not be handled by WoeEncoder
                # since np.nan will fail the equality check
                assert_all_finite(X[col])

            woe_value = woe(X[col], y,
                            conditional=col in self.conditional_cols,
                            na_values=self.na_values)
            self.mapping_[col] = woe_value
        return self

    def transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self, ['mapping_'])
        x = X.copy()
        for col in set(self.cols) & set(X.columns):
            if col not in self.mapping_:
                raise ValueError('Column {} not seen during the fit() process.'.format(col))

            if col not in self.conditional_cols:
                # TODO: Better way to deal with values that didn't appear in the fit() process
                # if value didn't appear in the fit() process find the nearest value for it
                # x[col] = x[col].map(lambda x: encode_with_nearest_key(self.mapping_[col], x))
                x[col] = x[col].map(self.mapping_[col]).fillna(0)
            else:
                # for conditional columns propagate the NaN values
                x[col] = x[col].map(self.mapping_[col])
        return x


if __name__ == '__main__':
    import random
    X = pd.DataFrame({'a': [random.randint(1, 3) for _ in range(10000)],
                      'b': [random.randint(1, 3) for _ in range(10000)],
                      'c': [random.randint(1, 3) for _ in range(10000)]})
    y = [int(random.random() > 0.5) for _ in range(10000)]

    WE = WoeEncoder(cols=['a', 'b'])
    encoded = WE.fit_transform(X, y)
    print(encoded)