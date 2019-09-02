"""
Light wrappers for imblearn oversampler.
Reference: https://github.com/scikit-learn-contrib/imbalanced-learn
"""
from imblearn.over_sampling import SMOTE as _SMOTE
from imblearn.over_sampling import BorderlineSMOTE as _BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE as _SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE as _KMeansSMOTE
from imblearn.over_sampling import RandomOverSampler as _RandomOverSampler


__all__ = ['SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', 'KMeansSMOTE', 'RandomOverSampler']


def _sampler_as_transformer(cls):
	""" Allow sampler from imblearn package to have a scikit-like interface (ie, with transform method) """
	transform_method = getattr(cls, 'fit_resample', None)
	setattr(cls, 'transform', transform_method)
	return cls


SMOTE = _sampler_as_transformer(_SMOTE)
BorderlineSMOTE = _sampler_as_transformer(_BorderlineSMOTE)
SVMSMOTE = _sampler_as_transformer(_SVMSMOTE)
KMeansSMOTE = _sampler_as_transformer(_KMeansSMOTE)
RandomOverSampler = _sampler_as_transformer(_RandomOverSampler)

