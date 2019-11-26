"""
Provide some dummy models to set up the baseline
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


__all__ = ['GroupByRegressor', 
		   'GroupByClassifier',
		   'RandomRegressor',
		   'RandomClassifier']


def _mode(x):
	""" Return the mode of the Series """
	return pd.Series(x).value_counts().index.tolist()[0]


class ProbabilityMixIn:

	def predict_proba(self, X):
		pred = np.empty((len(X), 2))
		pred[:, 1] = self.predict(X)
		pred[:, 0] = 1 - pred[:, 1]
		return pred


class RandomEstimator(BaseEstimator):

	def __init__(self, random_state=None):
		self.random_state = random_state

	def fit(self, X, y, sample_weight=None):
		return self

	def predict(self, X):
		np.random.seed(self.random_state)
		return np.random.random(size=(len(X)))


class GroupByEstimator(BaseEstimator):

	def __init__(self, strategy='mean'):
		self.strategy = strategy

	def _to_frame(self, X):
		""" Convert a numpy array to pd.DataFrame """
		if isinstance(X, pd.DataFrame):
			X_cols = X.columns.tolist()
		else:
			X_cols = ['f{}'.format(i) for i in range(X.shape[1])]
			X = pd.DataFrame(X, columns=X_cols)
		return X, X_cols

	def fit(self, X, y, sample_weight=None):
		# easier to work with dataframes
		X, X_cols = self._to_frame(X)
		y = pd.Series(y, name='target')

		df = pd.concat([X, y], axis=1)

		if self.strategy == 'mean':
			self.stats = (df.groupby(X_cols)['target'].mean(), df['target'].mean())
		elif self.strategy == 'median':
			self.stats = (df.groupby(X_cols)['target'].median(), df['target'].median())
		elif self.strategy == 'mode':
			self.stats = (df.groupby(X_cols)['target'].apply(_mode), _mode(df['target']))
		else:
			raise ValueError('Only mean, mode and median is supported.')

		return self

	def predict(self, X):
		stats, fill = self.stats
		X, X_cols = self._to_frame(X)
		pred = X.merge(stats, how='left', left_on=X_cols, right_index=True)['target']

		# there can be missings
		return pred.fillna(fill).values


class GroupByRegressor(GroupByEstimator, RegressorMixin):
	pass

class GroupByClassifier(GroupByEstimator, ProbabilityMixIn, ClassifierMixin):
	pass

class RandomRegressor(RandomEstimator, RegressorMixin):
	pass

class RandomClassifier(RandomEstimator, ProbabilityMixIn, ClassifierMixin):
	pass
