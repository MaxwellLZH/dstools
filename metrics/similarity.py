import numpy as np


def cosine_similarity(v1, v2):
    """ Cosine similartiy between two vector"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def jarccard_index(A, B, ignore_na=True):
    """ Calculate the jarccard index (aka jaccard similarity).
    :param ignore_na: Whether to filter out the missing value before calculation
    """
    def to_nonmissing_set(X):
        from sklearn.utils import is_scalar_nan
        return set(filter(lambda x: not is_scalar_nan(x), X))
    if ignore_na:
        A, B = to_nonmissing_set(A), to_nonmissing_set(B)
    else:
        A, B = set(A), set(B)
    return len(A & B) / len(A | B)


# alias
jaccard_similarity = jarccard_index


def calculate_relationship(x, y):
    """ Determine if y is positive|negative|unrelated with x"""
    import statsmodels.api as sm

    x = sm.add_constant(x)
    model = sm.OLS(y, x, missing='drop')
    model_result = model.fit()
    coef = model_result.params[1]
    if coef > 0:
        return 'postive'
    elif coef == 0:
        return 'unrelated'
    else:
        return 'negative'

