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


