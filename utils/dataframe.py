from itertools import groupby

from .misc import flatten_list


def create_multilevel_index(index_names, top_level_index_fn, bottom_level_index_fn, names=None):
    """ Create two-level multilevel index from given index names.
    :param top_level_index_fn: Function for getting the top level index
    :param bottom_level_index_fn: Function for getting the bottom level index
    """
    levels = groupby(index_names, top_level_index_fn)
    grouped_index = [(i[0], [bottom_level_index_fn(j) for j in i[1]]) for i in levels]
    top_level_index = [[i] * len(j) for i, j in grouped_index]
    bottom_level_index = [j for _, j in grouped_index]
    # concatenate the top and bottom level index into a sinlge flat list
    top_level_index = flatten_list(top_level_index)
    bottom_level_index = flatten_list(bottom_level_index)
    return pd.MultiIndex.from_arrays([top_level_index, bottom_level_index], names=names)