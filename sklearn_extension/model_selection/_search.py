from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._search import BaseSearchCV
from collections.abc import Mapping, Iterable
from itertools import product
from functools import partial, reduce
import operator
import numpy as np
import weakref


def _infer_dtype(array):
    """ Infer the data types from an array """
    if hasattr(array, 'dtype'):
        return array.dtype.type
    return type(array[0])


def _convert_dtype(num, dtype):
    """ Round integer and keep float number as it is """
    if dtype in (np.int, np.int0, np.int8, np.int16, np.int32, np.intp):
        return dtype(np.round(num, 0))
    else:
        return dtype(num)


def as_sorted_array(array):
    if hasattr(array, 'dtype'):
        array.sort()
        return array
    else:
        dtype = _infer_dtype(array)
        return np.array(sorted(array), dtype=dtype)


def _infer_interpolation(array):
    """ Infer the interpolation of an array """
    pass


_constraint_functions = {
    'positive': lambda x: x > 0,
    'nonnegative': lambda x: x >= 0
}


def get_edge_grid(array, target, interpolate='linear', constraint='positive'):
    """" Note: The array should be sorted """
    dtype = _infer_dtype(array)
    f_constraint = _constraint_functions.get(constraint.lower(), lambda _: True)

    if target == array.min():
        side = 0
        cand, right = array[0], array[1]
        if interpolate == 'linear':
            return np.array(
                list(filter(f_constraint, [cand - 2 * (right - cand), cand - (right - cand), cand])),
                dtype=dtype), side
        else:
            raise ValueError('Only linear interpolation is supported for now')

    elif target == array.max():
        side = 1
        cand, left = array[-1], array[-2]
        if interpolate == 'linear':
            return np.array(
                list(filter(f_constraint, [cand, cand + (cand - left), cand + 2 * (cand - left)])),
                dtype=dtype), side
        else:
            raise ValueError('Only linear interpolation is supported for now')

    else:
        return np.array([target], dtype=dtype), -1


def get_finer_grid(array, cand, interpolate='linear'):
    """ Note: The array should be sorted """
    dtype = _infer_dtype(array)
    if len(array) < 3:
        return np.array([cand], dtype=dtype)

    cand_idx = np.where(array==cand)[0][0]
    left, right = array[cand_idx-1], array[cand_idx+1]

    if interpolate == 'linear':
        next_grid = [cand]

        multiplier = 0.5
        while 1:
            left_cand = _convert_dtype(cand - multiplier * (cand - left), dtype)
            if left_cand == cand:
                # no possible finer grid
                break
            if left_cand not in array:
                next_grid.append(left_cand)
                break
            multiplier /= 2

        multiplier = 0.5
        while 1:
            right_cand = _convert_dtype(cand + multiplier * (right - cand), dtype)
            if right_cand == cand:
                # no possible finder grid
                break
            if right_cand not in array:
                next_grid.append(right_cand)
                break
            multiplier /= 2
    else:
        raise ValueError('Only linear interpolation is supported for now')
    return np.array(sorted(next_grid), dtype=dtype)


class ExpandingParamGrid(object):
    """ A parameter grid that can be expanded based on the evaluation result
        The grid is a regular Python dictionary except that it looks for the
        `_expand_`, `_constrain_`,  `_finer_` keyword in the dictionary in the dictionary.

        `_expand_` is a boolean value indicating whether the grid should be expanded
        when the edge value turned out to be the best choice, default is True
        `_finer_` should be an integer value, it determines that
        maximum number of times to expand if there's still choice left. Default expand 1 time.
        `_constrain_` is used to filter out any new candidate parameters that doesn't meet a certain
        criteria, current supported contrains are ['positive', 'nonnegative']
    """
    def __init__(self, param_grid, expand_edge=True, finer_grid=1):
        """
        :param expand_edge: Whether to add extra grid, when the returned best parameter is on the edge of the grid
        :param finer_grid: Try to find finer grid
        """
        if not isinstance(param_grid, dict):
            raise ValueError('The grid should be a dictionary')
        self.expand_edge = param_grid.get('_expand_', expand_edge)
        # converts to 1 when provided as Boolean
        self.finer_grid = int(param_grid.get('_finer_', finer_grid))
        self.constraint = param_grid.get('_constrain_', 'positive')
        # self.side keeps track of the edge grid that have already been searched
        # it's a dictionary mapping parameter to its side
        # -1 => previous search is not an edge case 0 => expanded on the left edge 1 => right edge
        self.side = param_grid.get('_side_', {k: -1 for k in param_grid.keys()})
        # self.base keeps track of the original parameter grid for getting the finer grid
        self.base = param_grid.get('_base_', param_grid)
        # grid is firstly expanded on the edge case then look for finer grid
        self.stage = param_grid.get('_stage_', 'expand')
        self.param_grid = {k: as_sorted_array(v) for k, v in param_grid.items() \
                           if not k.startswith('_')}

    def get_expanded_edge(self, best_params):
        # get the expanded edge for each field and search the combinations
        next_grid = dict()
        grid = self.param_grid
        prev_side = self.side
        cur_side = self.side.copy()
        base = self.base.copy()

        for field in grid.keys():
            if len(grid[field]) == 1:
                next_grid[field] = grid[field]
            else:
                print(grid, best_params)
                candidates, side = get_edge_grid(grid[field], best_params[field],
                                           interpolate='linear', constraint=self.constraint)
                # check if the same edge parameter remains the best one
                if len(candidates) > 1 and side != -1 and side + prev_side[field] != 1:
                    next_grid[field] = candidates
                    cur_side[field] = side
                    base[field] = candidates

        if next_grid:
            next_grid.update({'_constrain_': self.constraint,
                              '_finer_': self.finer_grid,
                              '_expand_': self.expand_edge,
                              '_stage_': self.stage,
                              '_side_': cur_side,
                              '_base_': base})
            return ExpandingParamGrid(next_grid)
        else:
            if self.finer_grid > 0:
                # starting the finer grid search
                next_grid = base.copy()
                next_grid.update({'_finer_': self.finer_grid,
                                  '_stage_': 'finer'})
                return ExpandingParamGrid(next_grid)
            else:
                return None

    def get_finer_grid(self, best_params):
        # get the finer grid for each field and search the combinations
        next_grid = dict()
        grid = self.param_grid

        if self.finer_grid < 1:
            return None

        for field in grid.keys():
            candidates = get_finer_grid(grid[field], best_params[field], interpolate='linear')
            if len(candidates) > 1:
                next_grid[field] = candidates
        if next_grid:
            next_grid.update({'_stage_': 'finer',
                              '_finer_': self.finer_grid - 1,
                              '_expand_': self.expand_edge})
            return ExpandingParamGrid(next_grid)
        else:
            return None

    def expand(self, best_params: dict):
        if self.stage == 'expand':
            return self.get_expanded_edge(best_params)
        else:
            return self.get_finer_grid(best_params)

    def __iter__(self):
        """ Copied from ParamGrid"""
        keys, values = zip(*self.param_grid.items())
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params

    def __len__(self):
        """ Copied from ParamGrid
            Note that it's only an appoximation
        """
        product = partial(reduce, operator.mul)
        return product(len(v) for v in self.param_grid.values())

    def __bool__(self):
        return bool(self.param_grid)


class SpecialRound(object):
    """  A placeholder for a single fitting round with special purpose such as resetting
        the learning rate or resetting the number of estimators etc
    """
    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __repr__(self):
        return self.__class__.__name__


class EarlyStoppingRound(SpecialRound):
    """ A placeholder for resetting the n_estimators with early stopping rounds """

    def __init__(self, early_stopping_rounds=30, eval_set=None, eval_metric='auc'):
        if eval_set is None:
            raise ValueError('Must provide evaluation set and evaluation metric.')
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_set = eval_set
        self.eval_metric = eval_metric


class OrderedParameterGrid(object):

    def __init__(self, param_grid, expand_edge=True, finer_grid=True):
        """
        :param expand_edge: Whether to add extra grid, when the returned best parameter is on the edge of the grid
        :param finer_grid: Try to find finer grid
        """
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                            'a list ({!r})'.format(param_grid))

        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]

        self.param_grid = [grid if isinstance(grid, SpecialRound) \
                            else ExpandingParamGrid(grid, expand_edge=expand_edge, finer_grid=finer_grid) \
                            for grid in param_grid]

    def iterate_next_grid(self):
        if self.param_grid:
            return self.param_grid.pop(0)

    def complete(self):
        """ Indicating that all the searches have been done """
        return not bool(self.param_grid)

    def expand(self, grid: ExpandingParamGrid, best_params: dict):
        # Try expand edge before going for the finer grid
        candidates = grid.expand(best_params)
        if candidates:
            # TODO: use deque
            print('{} expanded to {}'.format(grid.stage, candidates.param_grid))
            self.param_grid.insert(0, candidates)

    def __len__(self):
        return sum(len(p) for p in self.param_grid)


class OrderedGridSearchCV(BaseSearchCV):

    def __init__(self, estimator, param_grid, expand_edge=True, finer_grid=True, scoring=None, fit_params=None,
                 n_jobs=None, iid='warn', refit=True, cv='warn', verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise-deprecating',
                 return_train_score='warn'):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.fit_params = fit_params
        # keep track of the best parameters as we go
        self.best_params_ = dict()
        self.param_grid = OrderedParameterGrid(param_grid, expand_edge=expand_edge, finer_grid=finer_grid)

    def fit(self, X, y=None, groups=None, **fit_params):
        """ Overwrite the fit method so we can store the dataset and use it in reset_n_tree() """
        self.X_ = weakref.proxy(X)
        self.y_ = weakref.proxy(y)
        super().fit(X, y, groups, **fit_params)

    def reset_n_tree(self, e):
        """ Reset the n_estimators parameter using the early_stopping_rounds """
        self.estimator.set_params(**{'n_estimators': 1000})
        self.estimator.fit(self.X_, self.y_, 
                           eval_set=e.eval_set,
                           eval_metric=e.eval_metric,
                           early_stopping_rounds=e.early_stopping_rounds,
                           verbose=False)
        self.best_params_['n_estimators'] = self.estimator.best_ntree_limit

    def _run_search(self, evaluate_candidates):
        while not self.param_grid.complete():
            next_grid = self.param_grid.iterate_next_grid()
            if isinstance(next_grid, EarlyStoppingRound):
                print('Resetting n_estimators with early stopping rule.')
                self.reset_n_tree(next_grid)
            else:
                # update the grid with the best parameters so far
                updated_grid = [{**self.best_params_, **grid} for grid in next_grid]
                num_grid = len(updated_grid)

                print('Tuning parameters for {}'.format(updated_grid))
                results = evaluate_candidates(updated_grid)
                best_index = results['rank_test_score'][-num_grid:].argmin()
                best_params = results['params'][-num_grid:][best_index]

                # update best parameters and expand the grids
                print('Best params: {}'.format(best_params))
                self.best_params_.update(best_params)
                self.param_grid.expand(next_grid, best_params)



if __name__ == '__main__':

    from sklearn.datasets import load_iris
    import xgboost as xgb
    import random

    iris = load_iris()
    X = iris.data
    y = np.array([int(random.random() < 0.4) for i in iris.target], dtype=int)

    param_grid = [
            {'max_depth': [2, 3, 4], 'min_child_weight': [1, 2]},
            {'gamma': np.array([0, 0.1, 0.2], dtype=float)},
            EarlyStoppingRound(30, eval_set=[(X, y)], eval_metric='auc'),
            {'subsample': [0.8, 0.9, 1.0]}
        ]

    OGS = OrderedGridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param_grid, cv=3, n_jobs=1)
    OGS.fit(X, y)
    print(OGS.cv_results_)
