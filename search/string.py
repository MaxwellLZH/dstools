import itertools
from operator import itemgetter


def build_prefix_tree(lst):
    """ Build a prefix tree, which is basically a nested dictionary """
    lst = sorted(lst)
    mapping = dict()
    
    for prefix, values in itertools.groupby(lst, itemgetter(0)):
        # convert iterator into a list
        values = list(values)
        if len(values) == 1:
            mapping[prefix] = values[0][1:]
        else:
            mapping[prefix] = build_prefix_tree([i[1:] for i in values])
    return mapping


def find_choices(tree, x):
    """ Find all the string with the common prefix """
    if isinstance(tree, str):
        return [x+tree]
    else:
        return [choice for p in tree for choice in find_choices(tree[p], x+p)]


def search_prefix_tree(tree, x):
    res = ''
    for ch in x:
        if ch not in tree:
            raise ValueError('{} does not exist in the prefix tree'.format(res+ch))
        tree = tree[ch]
        res += ch
        
        if isinstance(tree, str):
            return res + tree
    raise ValueError('Multiple matches found for {}: {}'.format(res, find_choices(tree, res)))
    

def search_prefix(lst, x):
    """ Search for elements with a given prefix by building a prefix tree"""
    tree = build_prefix_tree(lst)
    return search_prefix_tree(tree, x)
    
