from functools import total_ordering
import numpy as np
import operator

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from pandas.api.types import is_numeric_dtype
import warnings


@total_ordering
class Edge:
    __slots__ = ['closed', 'value']
    def __init__(self, value, closed):
        self.value = value
        self.closed = closed
        
    def __lt__(self, other):
        return self.value < other.value
    
    def __gt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value and self.closed == other.closed
    
    def __repr__(self):
        return 'Edge<value={}, closed={}>'.format(self.value, self.closed)


def find_edge(e1, e2, edge='left', expand=True):
    """ Return the appropriate edge given side and action """
    # the operator that should return e1
    if (edge == 'left') != expand:
        op1, op2 = operator.gt, operator.lt
    else:
        op1, op2 = operator.lt, operator.gt
    
    closed = e1.closed or e2.closed if expand else e1.closed and e2.closed
    if op1(e1, e2):
        return e1
    elif op2(e1, e2):
        return e2
    else:
        return Edge(e1.value, closed)


class Interval:
    def __init__(self, left=-np.inf, right=np.inf, left_closed=False, right_closed=False):
        self.left = Edge(left, left_closed)
        self.right = Edge(right, right_closed)
        
    def __repr__(self):
        return '{lb}{l}, {r}{rb}'.format(lb='[' if self.left.closed else '(',
                                                l=self.left.value,
                                                r=self.right.value,
                                                rb=']' if self.right.closed else ')')
    @classmethod
    def from_edge(cls, left, right):
        return cls(left.value, right.value, left.closed, right.closed)
    
    def copy(self):
        return Interval.from_edge(self.left, self.right)
        
    def __eq__(self, other):
        return self.left == other.left and self.right == other.right
    
    def __and__(self, other):
        return Interval.from_edge(find_edge(self.left, other.left, edge='left', expand=False), 
                                  find_edge(self.right, other.right, edge='right', expand=False))
    
    def __or__(self, other):
        return Interval.from_edge(find_edge(self.left, other.left, edge='left', expand=True), 
                                  find_edge(self.right, other.right, edge='right', expand=True))
    
    def clip_right(self, value, closed=True):
        assert value > self.left.value
        right = Edge(value, closed)
        right = find_edge(self.right, right, edge='right', expand=False)
        return Interval.from_edge(self.left, right)

    def clip_left(self, value, closed=True):
        assert value < self.right.value
        left = Edge(value, closed)
        left = find_edge(self.left, left, edge='left', expand=False)
        return Interval.from_edge(left, self.right)

    def expand_right(self, value, closed=True):
        right = Edge(value, closed)
        right = find_edge(self.right, right, edge='right', expand=True)
        return Interval.from_edge(self.left, right)
    
    def expand_left(self, value, closed=True):
        left = Edge(value, closed)
        self.left = find_edge(self.left, left, edge='left', expand=True)
        return Interval.from_edge(left, self.right)



def parse_tree(tree):
    """ Parse a tree object into a list of intervals, where intervals[i] is the 
        interval for node[i] in this tree
    """
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
        intervals[left] = intervals[node_idx].clip_right(threshold, closed=True)
        intervals[right] = intervals[node_idx].clip_left(threshold, closed=False)
    
    return intervals


class TreeBinner(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols=None, bins=10, categorical_cols=None, min_frac=0.05, fill=-1, random_state=1024):
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
        self.fill = fill
        
        self.discrete_encoding = None
        self.trees = None
        self.interval_mapping = None
        self.random_state = random_state
    
    def _fit(self, X: pd.Series, y): 
        if not is_numeric_dtype(X) and X.name not in self.categorical_cols:
            raise ValueError('Column {} is not numeric and not in categorical_cols.'.format(X.name))
        
        if X.name in self.categorical_cols:
            X = self.encode_with_label(X, y)
        
        _, X, y = drop_na(X, y, according='x')
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
        return X.fillna(self.fill).astype(int)
    
    def transform(self, X, y=None):
        if self.interval_mapping is None:
            raise NotFittedError('This {} is not fitted. Call the fit method first.'.format(self.__class__.__name__))
        
        X_ = X.copy()
        with warnings.catch_warnings():
            # Ignore the set on copy warnings
            warnings.simplefilter('ignore')
            for col in self.interval_mapping:
                X_[col] = self._transform(X_[col])
        return X_
