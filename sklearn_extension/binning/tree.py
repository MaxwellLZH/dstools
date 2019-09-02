from functools import total_ordering
import pandas as pd
import numpy as np
import operator

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from pandas.api.types import is_numeric_dtype
import warnings


@total_ordering
class Interval:
    """ A dummy inplementation of Interval, no validation on (and|or) operations and 
        doesn't include information on open or close edges.
    """
    
    def __init__(self, left=-np.inf, right=np.inf):
        self.left = left
        self.right = right
        
    def __repr__(self):
        return 'Interval<{}, {}>'.format(self.left, self.right)
    
    def __lt__(self, other):
        return self.left < other.left
    
    def __eq__(self, other):
        return (self.left == other.left) and (self.right == other.right)
    
    def copy(self):
        return Interval(self.left, self.right)
        
    def __eq__(self, other):
        return self.left == other.left and self.right == other.right
    
    def __and__(self, other):
        return Interval(max(self.left, other.left), min(self.right, other.right))
    
    def __or__(self, other):
        return Interval(min(self.left, other.left), max(self.right, other.right))
    
    def clip_right(self, value):
        return Interval(self.left, min(self.right, value))
    
    def clip_left(self, value):
        return Interval(max(self.left, value), self.right)

    def expand_right(self, value):
        return Interval(self.left, max(self.right, value))
    
    def expand_left(self, value):
        return Interval(min(self.left, value), self.right)


def parse_tree(tree):
    """ Parse a tree object into a list of intervals, where intervals[i] is the 
        interval for node[i] in this tree
    """
    if hasattr(tree, 'tree_'):
        return parse_tree(tree.tree_)
    
    children_left = tree.children_left
    children_right = tree.children_right
    # feature_index = tree.feature
    thresholds = tree.threshold

    intervals = [Interval() for _ in range(tree.node_count)]

    for node_idx in range(tree.node_count):
        # skip the leaf nodes
        if children_left[node_idx] == -1:
            continue

        left, right = children_left[node_idx], children_right[node_idx]
        threshold = thresholds[node_idx]
        intervals[left] = intervals[node_idx].clip_right(threshold)
        intervals[right] = intervals[node_idx].clip_left(threshold)
    
    return intervals


class TreeBinner(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 cols=None, 
                 bins=10, 
                 categorical_cols=None, 
                 min_frac=0.05, 
                 encode=True,
                 fill=-1, 
                 random_state=1024):
        """
        :param cols: A list of columns to perform binning, if set to None, perform binning on all columns.
        :param bins: Maximum number of bins to split into
        :param categorical_cols: A list of categorical columns
        :param min_frac: Minimum fraction of samples within each bin
        :param fill: Value used for inputing missing value
        :param random_state: Random state used for growing trees

        Usage:
        --------------
        >>> from sklearn.datasets import load_breast_cancer
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> TB = TreeBinner(bins=5, min_frac=0.05)
        >>> CB.fit(X, y)
        >>> CB.interval_mapping  # Get the interval for each bin
        """
        self.cols = cols
        self.categorical_cols = categorical_cols or []
        self.bins = bins
        self.min_frac = min_frac
        self.encode = encode
        self.fill = fill
        
        self.discrete_encoding = None
        # the min and max value for each feature
        self.min_ = None
        self.max_ = None
        
        self.trees = None
        self.interval_mapping = None
        self.random_state = random_state
        
    @staticmethod
    def _drop_na(X, y):
        # make sure y has the same index as X
        y = pd.Series(y, index=X.index)
        
        idx = X[X.notnull()].index
        return X.loc[idx], y.loc[idx]
    
    def _fit(self, X: pd.Series, y): 
        if not is_numeric_dtype(X) and X.name not in self.categorical_cols:
            raise ValueError('Column {} is not numeric and not in categorical_cols.'.format(X.name))
        
        if X.name in self.categorical_cols:
            X = self.encode_with_label(X, y)
        
        if not self.encode:
            self.min_[X.name], self.max_[X.name] = X.min(), X.max() 
            
        X, y = self._drop_na(X, y)
        DT = DecisionTreeClassifier(max_leaf_nodes=self.bins,
                                    min_samples_leaf=self.min_frac, 
                                    random_state=self.random_state)
        DT.fit(X.to_frame(), y)
        return parse_tree(DT.tree_), DT
    
    def encode_with_label(self, X: pd.Series, y):
        """ Encode categorical features with its percentage of positive samples"""
        y = pd.Series(y)
        pct_pos = y.groupby(X).mean()
        self.discrete_encoding[X.name] = pct_pos
        return X.map(pct_pos)
    
    def fit(self, X, y, **fit_params):
        cols = self.cols or X.columns
        
        if not self.encode:
            self.min_, self.max_ = {}, {}
        
        self.interval_mapping, self.trees, self.discrete_encoding = {}, {}, {}
        for col in cols:
            self.interval_mapping[col], self.trees[col] = self._fit(X[col], y)
        return self
    
    def _transform(self, X: pd.Series, y=None):
        col_name = X.name
        
        if col_name in self.categorical_cols:
            X = X.map(self.discrete_encoding[col_name])
        
        tree = self.trees[col_name]
        valid_index = X.notnull()
        X[valid_index] = tree.apply(X[valid_index].to_frame())
        
        if self.encode:
            return X.fillna(self.fill).astype(int)
        else:
            # create a mapping from interval index to edge value
            # edge left is only used for the leftmost node (where Interval.left == -np.inf)
            _min, _max = self.min_[col_name], self.max_[col_name]
            cutoff_mapping = {i: _min if j.left == -np.inf else min(_max, j.right)
                                for i, j in enumerate(self.interval_mapping[col_name])}
            X[valid_index] = X[valid_index].map(cutoff_mapping)
            # leave NaN unfilled if self.encode is False
            return X
    
    def transform(self, X, y=None):
        if self.interval_mapping is None:
            raise NotFittedError('This {} is not fitted. Call the fit method first.'.format(self.__class__.__name__))
        
        X_ = X.copy()
        with warnings.catch_warnings():
            # Ignore the set on copy warnings
            warnings.simplefilter('ignore')
            for col in self.cols or X.columns:
                X_[col] = self._transform(X_[col])
        return X_


def tree_binning(X: pd.Series, y, n: int, min_frac: float=0.05, encode: bool=True, fill: int=-1, random_state: int=1024):
    s_name = X.name
    TB = TreeBinner(bins=n, min_frac=min_frac, encode=encode, fill=fill, random_state=random_state)
    binned = TB.fit_transform(X.to_frame(), y)
    return binned[s_name], None if encode else sorted(binned[s_name].unique())
