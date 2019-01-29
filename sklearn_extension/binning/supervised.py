from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np

from ..utils import force_zero_one, make_series, searchsorted, assign_group
from ..binning.unsupervised import EqualFrequencyBinning, EqualWidthBinning


class ChiSquareBinning(BaseEstimator, TransformerMixin):

    def __init__(self, max_bin, categorical_cols=None, encode=True, fill=-1,
                 force_monotonic=True, force_mix_label=True,
                 strict=True, ignore_na=True, prebin=100):
        """
        :param max_bin: The number of bins to split into
        :param categorical_cols: A list of categorical columns
        :param encode: If set to False, the result of transform will be right cutoff point of the interval
        :param fill: Label for missing value
        :param force_monotonic:  Whether to force the bins to be monotonic with the
                positive proportion of the label
        :param force_mix_label:  Whether to force the all the bins to have mix labels
        :param strict: If set to True, equal values will not be treated as monotonic
        :param ignore_na: The monotonicity check will ignore missing value
        :param prebin: An integer, number of bins to split into before the chimerge process.
        """
        self.max_bin = max_bin
        self.categorical_cols = categorical_cols or []
        self.encode = encode
        self.fill = fill
        self.force_monotonic = force_monotonic
        self.force_mix_label = force_mix_label
        self.strict = strict
        self.ignore_na = ignore_na
        # A dictionary mapping column name to its encoding
        self.discrete_encoding = dict()
        # A dictionary mapping column name to its cutoff points
        self.bins = dict()

        self.prebin = prebin

        # self.keep_history = keep_history
        # self.history = list()  # a list of (Merge From, Merge to, Criteria)
        # self.history_summary = list()

        self._chisquare_cache = dict()

    def calculate_chisquare(self, X: pd.Series, y: pd.Series, expected_ratio: float) -> float:
        # try to get from the cache first
        unique_x = frozenset(X)
        if self._chisquare_cache.get(unique_x, False):
            return self._chisquare_cache[unique_x]

        summary = y.groupby(X).agg(['count', 'sum']).rename(columns={'sum': 'actual_pos'})
        summary['actual_neg'] = summary['count'] - summary['actual_pos']
        summary['expected_pos'] = summary['count'] * expected_ratio
        summary['expected_neg'] = summary['count'] - summary['expected_pos']
        chi2 = (summary['actual_pos'] - summary['expected_pos']) ** 2 / summary['expected_pos'] + \
               (summary['actual_neg'] - summary['expected_neg']) ** 2 / summary['expected_neg']
        dgfd = summary.shape[0] - 1
        chi2 = chi2.sum() / dgfd
        self._chisquare_cache[unique_x] = chi2
        return chi2

    @staticmethod
    def sorted_two_gram(X):
        """ Two gram with the left element smaller than the right element in each pair
            eg. sorted_two_gram([1, 3, 2]) -> [(1, 2), (2, 3)]
        """
        unique_values = np.unique(X[X.notnull()])
        return [(unique_values[i], unique_values[i + 1])
                for i in range(len(unique_values) - 1)]

    @staticmethod
    def is_monotonic(i, strict=True, ignore_na=True) -> bool:
        """ Check if an iterable is monotonic """
        i = make_series(i)
        diff = i.diff()[1:]
        if ignore_na:
            diff = diff[diff.notnull()]
        sign = diff > 0 if strict else diff >= 0
        if sign.sum() == 0 or (~sign).sum() == 0:
            return True
        return False

    @staticmethod
    def find_candidate(values: pd.Series, target) -> list:
        """ Return a list of candidatate values that's next bigger or next smaller than the target value.
            The candidate list will have only one element when the target is the min or max in X.
            ex. find_candidate([1, 2, 3, 0], 2) => [1, 3]
        """
        values = sorted(values)
        idx = values.index(target)
        cnt = len(values)
        return [values[i] for i in [idx - 1, idx + 1] if 0 <= i < cnt]

    # def maybe_save_summary(self, X, y):
    #     # save the summary table if needed
    #     if self.keep_history:
    #         self.history_summary.append(self.calculate_summary(X, y))

    def encode_with_label(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """ Encode categorical features with its percentage of positive samples"""
        pct_pos = y.groupby(X).mean()
        # save the mapping for transform()
        self.discrete_encoding[X.name] = pct_pos
        return X.map(pct_pos)

    def is_monotonic_post_bin(self, X, y):
        """ Check whether the proportion of positive label is monotonic to bin value"""
        summary = self.calculate_summary(X, y)
        pct_pos = summary.sort_values('value')['pct_pos']
        return self.is_monotonic(pct_pos, self.strict, self.ignore_na)

    def calculate_summary(self, X, y) -> pd.DataFrame:
        """ Return a summary dataframe that showes the
            1. chisquare value
            2. proportion of positive samples
            for each unique value in X, which will be used for merging bins
        """
        pct_pos = []
        uniq_values = X[X.notnull()].unique()
        for value in uniq_values:
            group_label = y[X == value]
            pct_pos.append(group_label.mean())

        return pd.DataFrame({'value': uniq_values, 'pct_pos': pct_pos})

    def merge_bin(self, X: pd.Series, replace_value, original_value) -> pd.Series:
        """ Replace the smaller value with the bigger one except when the replace value is the
            minimum of X, that case we replace the bigger value with the the smaller one.
        :param X: The original series.
        """
        if replace_value == X.min():
            return X.replace(original_value, replace_value)
        if replace_value < original_value:
            replace_value, original_value = original_value, replace_value
        return X.replace(original_value, replace_value)

    def merge_chisquare(self, X, y) -> pd.Series:
        """ Performs a single merge based on chi square value
            returns a new X' with new groups
        """
        expected_ratio = y.mean()
        candidate_pairs = self.sorted_two_gram(X)
        # find the pair with minimum chisquare in one-pass
        min_idx, min_chi2 = 0, np.inf
        for i, pair in enumerate(candidate_pairs):
            idx = X[X.isin(pair)].index
            chi2 = self.calculate_chisquare(X[idx], y[idx], expected_ratio)
            if chi2 < min_chi2:
                min_idx, min_chi2 = i, chi2

        # replace the smaller value with the bigger one except for the minimum pair
        small, large = candidate_pairs[min_idx]
        return self.merge_bin(X, small, large)

    def merge_purity(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """ Performs a single merge trying to merge bins with only 0 or a label into the adjacent mixed label bin
        """
        summary = self.calculate_summary(X, y)
        idx_single_label = summary['pct_pos'] * (1 - summary['pct_pos']) == 0
        idx_single_label = np.where(idx_single_label)[0]
        if len(idx_single_label) == 0:
            return X
        idx_single_label = idx_single_label[0]
        value_pure_bin = summary.ix[idx_single_label, 'value']

        merge_candidates = self.find_candidate(X[X.notnull()].unique(), value_pure_bin)
        # it merges to the candidate that has mix labels, if both candidates do, then
        # it will merge to the larger value, if neither does, if will merge both candidates
        if len(merge_candidates) == 1:
            return self.merge_bin(X, merge_candidates[0], value_pure_bin)
        else:
            left_cand, right_cand = merge_candidates
            pct_pure_bin = summary.loc[summary['value'] == value_pure_bin, 'pct_pos'].values[0]
            pct_pos_left = summary.loc[summary['value'] == left_cand, 'pct_pos'].values[0]
            pct_pos_right = summary.loc[summary['value'] == right_cand, 'pct_pos'].values[0]

            if 0 < (pct_pos_left + pct_pure_bin) / 2 < 1:
                return self.merge_bin(X, left_cand, value_pure_bin)
            elif 0 < (pct_pos_right + pct_pure_bin) / 2 < 1:
                return self.merge_bin(X, right_cand, value_pure_bin)
            else:
            # if both direction can result in mixed label or neither does
            # merge into the bin that results in smaller chisquare value
                idx_left = X[X.isin([value_pure_bin, left_cand])].index
                idx_right = X[X.isin([value_pure_bin, left_cand])].index
                expected_ratio = y.mean()
                chi2_left = self.calculate_chisquare(X[idx_left], y[idx_left], expected_ratio)
                chi2_right = self.calculate_chisquare(X[idx_right], y[idx_right], expected_ratio)

                if chi2_left < chi2_right:
                    return self.merge_bin(X, left_cand, value_pure_bin)
                else:
                    return self.merge_bin(X, right_cand, value_pure_bin)

    def _fit(self, X, y, **fit_parmas):
        """ Fit a single feature and return the cutoff points"""
        y = force_zero_one(y)
        X, y = make_series(X), make_series(y)
        # if X is discrete, encode with positive ratio in y
        if X.name in self.categorical_cols:
            X = self.encode_with_label(X, y)

        # the number of bins is the number of cutoff points minus 1
        n_bins = X.nunique() - 1
        # speed up the process with prebinning
        if self.prebin and n_bins > self.prebin:
            EFB = EqualFrequencyBinning(n=self.prebin, encode=False)
            X = EFB.fit_transform(X)
            X = make_series(X)

        n_bins = X.nunique() - 1
        # merge bins based on chi square
        while n_bins > self.max_bin:
            X = self.merge_chisquare(X, y)
            # TODO: replace nuique() with n_bins -= 1 ??
            n_bins = X.nunique() - 1

        # merge bins to create mixed label in every bin
        if self.force_mix_label and n_bins > 1:
            # loop until the number of bins doesn't change anymore
            prev_n_bins = n_bins
            while 1:
                X = self.merge_purity(X, y)
                if X.nunique() == prev_n_bins:
                    break
                prev_n_bins = X.nunique()

        # merge bins to keep bins to be monotonic
        if self.force_monotonic:
            while X.nunique() - 1 > 2 and not self.is_monotonic_post_bin(X, y):
                X = self.merge_chisquare(X, y)

        # note here we're adding the min_x and  largest number possible so
        # even if the transform() encounters values outside of range, we'll still be able to
        # encode them
        return np.unique(X[X.notnull()])

    def _transform(self, X: pd.Series, y=None):
        """ Transform a single feature"""
        if not self.bins:
            raise NotFittedError('This ChiSquareBinner is not fitted. Call the fit method first.')

        # map discrete value to the positive proportion
        col_name = X.name
        if col_name not in self.bins:
            raise ValueError('Column {} does\'t exist during fit().'.format(col_name))
        if col_name in self.categorical_cols:
            X = X.map(self.discrete_encoding[col_name])

        if self.encode:
            return searchsorted(self.bins[col_name], X, self.fill)
        else:
            return assign_group(X, self.bins[col_name])

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        """
        :param X: Pandas DataFrame with shape (n_sample, n_feature)
        :param y: a label column with shape (n_sample, )
        """
        # check if any of the string type columns is not in category_cols list
        _error_cols = set(X.select_dtypes('O').columns) - set(self.categorical_cols)
        if _error_cols:
            raise ValueError('The following columns are string but not included in the categorical_col: ' +
                             repr(list(_error_cols)))

        for col in X.columns:
            self.bins[col] = self._fit(X[col], y)
        return self

    def transform(self, X: pd.DataFrame, y=None):
        x = X.copy()
        for col in x.columns:
            x[col] = self._transform(x[col])
        return x


if __name__ == '__main__':
    import random
    import pickle
    import time

    X = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 6, 6, 6, 7, 7, 7, 7, 7]
    y = [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0]
    X = pd.DataFrame({'a': X, 'b': X, 'c': X})

    X = pd.DataFrame({'a': [random.randint(1, 20) for _ in range(10000)],
                   'b': [random.randint(1, 20) for _ in range(10000)],
                 'c': [random.randint(1, 20) for _ in range(10000)]})
    y = [int(random.random() > 0.5) for _ in range(10000)]

    CB = ChiSquareBinning(max_bin=5, categorical_cols=['a'], force_mix_label=False, force_monotonic=False,
                          prebin=100, encode=True, strict=False)

    start = time.time()
    CB.fit(X, y)
    print(time.time() - start)
    print(CB.transform(X).nunique())
    print(CB.bins)
    print(pd.concat([X, CB.transform(X)], axis=1))
