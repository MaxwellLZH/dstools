from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np
from typing import Dict, Iterable, Tuple

from ..utils import force_zero_one, make_series, searchsorted, assign_group
from ..binning.unsupervised import EqualFrequencyBinning


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

        self._chisquare_cache = dict()

    def calculate_chisquare(self, mapping: Dict[int, list], candidates: Iterable) -> float:
        # try to get from the cache first
        unique_x = frozenset(candidates)
        if self._chisquare_cache.get(unique_x, False):
            return self._chisquare_cache[unique_x]

        count = {k: len(v) for k, v in mapping.items()}
        actual_pos = {k: sum(v) for k, v in mapping.items()}
        actual_neg = {k: (count[k] - actual_pos[k]) for k in candidates}
        expected_ratio = sum(actual_pos.values()) / sum(count.values())
        expected_pos = {k: v * expected_ratio for k, v in count.items()}
        expected_neg = {k: (count[k] - expected_pos[k]) for k in candidates}

        chi2 = sum((actual_pos[k] - expected_pos[k])**2 / expected_pos[k] + \
                   (actual_neg[k] - expected_neg[k])**2 / expected_neg[k]
                   for k in candidates)
        dgfd = len(candidates) - 1
        chi2 = chi2 / dgfd
        self._chisquare_cache[unique_x] = chi2
        return chi2

    @staticmethod
    def sorted_two_gram(X):
        """ Two gram with the left element smaller than the right element in each pair
            eg. sorted_two_gram([1, 3, 2]) -> [(1, 2), (2, 3)]
        """
        unique_values = sorted(X)
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
    def find_candidate(values: Iterable, target: int) -> list:
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
        X, y = make_series(X), make_series(y)
        pct_pos = y.groupby(X).mean()
        # save the mapping for transform()
        self.discrete_encoding[X.name] = pct_pos
        return X.map(pct_pos)

    def is_monotonic_post_bin(self, mapping: Dict[int, list]):
        """ Check whether the proportion of positive label is monotonic to bin value"""
        pct_pos = sorted([(k, np.mean(v)) for k, v in mapping.items()])
        return self.is_monotonic([i[1] for i in pct_pos], self.strict, self.ignore_na)

    # def calculate_summary(self, X, y) -> pd.DataFrame:
    #     """ Return a summary dataframe that showes the
    #         1. chisquare value
    #         2. proportion of positive samples
    #         for each unique value in X, which will be used for merging bins
    #     """
    #     pct_pos = []
    #     uniq_values = X[X.notnull()].unique()
    #     for value in uniq_values:
    #         group_label = y[X == value]
    #         pct_pos.append(group_label.mean())
    #     return pd.DataFrame({'value': uniq_values, 'pct_pos': pct_pos})

    def merge_bin(self, mapping: Dict[int, list], replace_value, original_value) -> Dict[int, list]:
        """ Replace the smaller value with the bigger one except when the replace value is the
            minimum of X, that case we replace the bigger value with the the smaller one.
        """
        minimum = min(mapping)

        def _replace(mapping: Dict[int, list], to_replace: int, value: int):
            mapping[value].extend(mapping[to_replace])
            del mapping[to_replace]
            return mapping

        if replace_value == minimum:
            return _replace(mapping, original_value, replace_value)

        # make sure replace_value is the bigger one
        if replace_value < original_value:
            replace_value, original_value = original_value, replace_value
        return _replace(mapping, original_value, replace_value)

    def merge_chisquare(self, mapping: Dict[int, list], candidates=None) -> Dict[int, list]:
        """ Performs a single merge based on chi square value
            returns a new X' with new groups
        :param candidates: the candidate values that are allowed to merge, default set to all the values
        """
        candidates = candidates or mapping.keys()
        candidate_pairs = self.sorted_two_gram(candidates)

        # find the pair with minimum chisquare in one-pass
        min_idx, min_chi2 = 0, np.inf
        for i, pair in enumerate(candidate_pairs):
            chi2 = self.calculate_chisquare(mapping, pair)
            if chi2 < min_chi2:
                min_idx, min_chi2 = i, chi2

        # replace the smaller value with the bigger one except for the minimum pair
        small, large = candidate_pairs[min_idx]
        return self.merge_bin(mapping, small, large)

    def merge_purity(self, mapping: Dict[int, list]) -> Tuple[Dict[int, list], bool]:
        """ Performs a single merge trying to merge bins with only 0 or a label into the adjacent mixed label bin
            Return the updated mapping and a purity label
        """
        # convert to list so we don't get error modifying dictionary during loop
        for k in sorted(list(mapping.keys())):
            pct_pos = np.mean(mapping[k])

            if pct_pos * (1 - pct_pos) == 0:
                merge_candidates = self.find_candidate(mapping.keys(), k)

                # it merges to the candidate that has mix labels, if both candidates do, then
                # it will merge to the larger value, if neither does, if will merge both candidates
                if len(merge_candidates) == 1:
                    return self.merge_bin(mapping, merge_candidates[0], k), False
                else:
                    left_cand, right_cand = merge_candidates
                    pct_pos_left = np.mean(mapping[left_cand])
                    pct_pos_right = np.mean(mapping[right_cand])

                    can_merge_left = 0 < (pct_pos_left + pct_pos) / 2 < 1
                    can_merge_right = 0 < (pct_pos_right + pct_pos) / 2 < 1

                    if can_merge_left and can_merge_right:
                        return self.merge_chisquare(mapping, [left_cand, k, right_cand]), False
                    elif can_merge_left:
                        return self.merge_bin(mapping, left_cand, k), False
                    elif can_merge_right:
                        return self.merge_bin(mapping, right_cand, k), False
                    else:
                        mapping = self.merge_bin(mapping, left_cand, k)
                        return self.merge_bin(mapping, right_cand, k), False
        else:
            return mapping, True

    def _fit(self, X, y, **fit_parmas):
        """ Fit a single feature and return the cutoff points"""
        y = force_zero_one(y)
        y = make_series(y)

        # if X is discrete, encode with positive ratio in y
        if X.name in self.categorical_cols:
            X = self.encode_with_label(X, y)

        # the number of bins is the number of cutoff points minus 1
        n_bins = X.nunique() - 1
        # speed up the process with prebinning
        if self.prebin and n_bins > self.prebin:
            EFB = EqualFrequencyBinning(n=self.prebin, encode=False)
            X = EFB.fit_transform(X)
            # X = make_series(X)

        # convert to mapping
        mapping = y.groupby(X).apply(list).to_dict()

        n_bins = len(mapping) - 1
        # merge bins based on chi square
        while n_bins > self.max_bin:
            mapping = self.merge_chisquare(mapping)
            n_bins = len(mapping) - 1

        # merge bins to create mixed label in every bin
        if self.force_mix_label and n_bins > 1:
            is_pure = False
            while not is_pure:
                mapping, is_pure = self.merge_purity(mapping)

        # merge bins to keep bins to be monotonic
        if self.force_monotonic:
            while len(mapping) - 1 > 2 and not self.is_monotonic_post_bin(mapping):
                mapping = self.merge_chisquare(mapping)

        return sorted(mapping.keys())

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

    X = pd.DataFrame({'a': [random.randint(1, 20) for _ in range(1000)],
                   'b': [random.randint(1, 20) for _ in range(1000)],
                 'c': [random.randint(1, 20) for _ in range(1000)]})
    y = [int(random.random() > 0.5) for _ in range(1000)]

    CB = ChiSquareBinning(max_bin=5, categorical_cols=['a'], force_mix_label=False, force_monotonic=False,
                          prebin=100, encode=True, strict=False)

    start = time.time()
    CB.fit(X, y)
    print(time.time() - start)
    print(CB.transform(X).nunique())
    print(CB.bins)
    print(pd.concat([X, CB.transform(X)], axis=1))
