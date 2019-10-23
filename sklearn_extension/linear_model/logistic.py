from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import warnings
import statsmodels.api as sm
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import sort_columns_logistic, sort_columns_tree


__all__ = ['StepwiseLogisticRegression', 'IncrementalLogisticRegression']


class StepwiseBase(BaseEstimator, ClassifierMixin):
    """ Base class for Stepwise LogisticRegression Model
        Subclasses should implement the following methods:

        add_feature(fit_results: List[(str, LogitResults)]) -> str
    
        drop_features(logit_result: LogitResults) -> List[str]
    """
    def __init__(self, 
                 cols=None,
                 max_feature=None,
                 force_positive_coef=True,
                 mode='forward', 
                 method='bfgs', 
                 refit=True,
                 **kwargs):
        """
        :param cols: columns ranked by importance from high to low, ex. iv
        :param mode: 'forward', 'backward' or 'bidirectional'
        :param force_positive_coef: Whether the coefficients should be all positive
        :param kwargs: keyword arguments for fitting the LogisticRegression model in 
            fit() method
        """
        if mode not in ('forward', 'backward', 'bidirectional'):
            raise ValueError('Only support forward, backward and bidirectional mode.')
        self.cols = cols
        self.max_feature = max_feature
        self.force_positive_coef = force_positive_coef
        self.mode = mode
        self.method = method
        self.refit = refit
        self.fit_kwargs = kwargs
        
        self.model_cols = None
        self.history = None
        self.model = None

    def add_feature(self, fit_results):
        raise NotImplementedError

    def drop_features(self, logit_result):
        raise NotImplementedError

    def _fit_logistic(self, X, y):
        logit_model = sm.Logit(y, X)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                logit_result = logit_model.fit(disp=False, method=self.method)
            return logit_result
        except np.linalg.LinAlgError:
            """ usually caused by multi-colinearity, simply ignore the column """
            return None

    def _check_param(self, X, y, model_cols):
        """ Check if there's any feature whose coefficients violates the constraint """
        if not self.force_positive_coef:
            return model_cols

        logit_result = self._fit_logistic(X[model_cols], y)
        coef = logit_result.params
        keep_cols = [c for c in model_cols if (coef[c] > 0 or c == 'const')]

        for c in model_cols:
            if c not in keep_cols:
                self.history.append(('Remove', c, 'violate constraint'))

        return keep_cols

    def forward_step(self, X, y, candidate_cols, model_cols):
        """ Perform a single forward step
            Return (candidate columns, model columns, flag)
            `flag` indicating whether the current step should be the last step
        """
        # prepare the `fit_results`, where each of them is a pair of column name and LogitResults
        fit_results = list()
        for cand in candidate_cols:
            if cand in model_cols:
                continue

            logit_result = self._fit_logistic(X[model_cols+[cand]], y)
            if logit_result is None:
                continue

            fit_results.append((cand, logit_result))

        add_col = self.add_feature(fit_results)

        if add_col is None:
            return candidate_cols, model_cols, True
        else:
            self.history.append(('Add', add_col, 'forward step'))
            candidate_cols = [c for c in candidate_cols if c != add_col]
            return candidate_cols, model_cols + [add_col], len(candidate_cols) == 0
        
    def backward_step(self, X, y, model_cols):
        """
            Return (model columns, flag)
            `flag` indicating whether the current step should be the last step
        """
        logit_result = self._fit_logistic(X[model_cols], y)
        # TODO: Better handling when a LinAlgError is encountered
        if logit_result is None:
            warnings.warn('A LinAlgError is encountered during the backward step.')
            return model_cols, True

        drop_cols = self.drop_features(logit_result)
        model_cols = [c for c in model_cols if c not in drop_cols]
        if drop_cols:
            self.history.append(('Remove', drop_cols, 'backward step'))
        return model_cols, len(drop_cols) == 0

    def fit(self, X, y, **fit_params):
        cols = self.cols or X.columns.tolist()
        self.history = list()
        X = sm.add_constant(X.copy())
        
        need_forward_step = self.mode in ('forward', 'bidirectional')
        need_backward_step = self.mode in ('bidirectional', 'backward')
        
        if need_forward_step:
            candidate_cols, model_cols = cols, ['const']
        else:
            candidate_cols, model_cols = None, cols + ['const']
        
        stop_sign = False
        # ignore the convergence warnings for now
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with tqdm() as pbar:

                while not stop_sign:
                    if need_forward_step:
                        candidate_cols, model_cols, forward_stop_sign = \
                                self.forward_step(X, y, candidate_cols, model_cols)
                    if need_backward_step:
                        model_cols, backward_stop_sign = self.backward_step(X, y, model_cols)

                    # check coefficients
                    model_cols = self._check_param(X, y, model_cols)

                    if self.mode == 'forward':
                        stop_sign = forward_stop_sign
                    elif self.mode == 'bidirectional':
                        stop_sign = forward_stop_sign and backward_stop_sign
                    else:
                        stop_sign = backward_stop_sign

                    pbar.update()

                    # Early stopping when max_feature is specified
                    if self.max_feature is not None:
                        n_model_cols = len([c for c in model_cols if c != 'const'])
                        if self.mode in ('forward', 'bidirectional'):
                            early_stop_sign = n_model_cols >= self.max_feature
                        else:
                            early_stop_sign = n_model_cols <= self.max_feature

                        stop_sign = stop_sign or early_stop_sign

        if 'const' in model_cols:
            model_cols.remove('const')
        self.model_cols = model_cols
        
        if self.refit:
            from sklearn.linear_model import LogisticRegression
            
            if len(self.model_cols) == 0:
                raise ValueError('No column was selected.')
            LR = LogisticRegression(**self.fit_kwargs)
            LR.fit(X[model_cols], y)
            self.model = LR
        return self

    def predict(self, X):
        if self.model is None:
            raise NotFittedError('The {} is not fitted yet.'.format(self.__class__))
        return self.model.predict(X[self.model_cols])
    
    def predict_proba(self, X):
        if self.model is None:
            raise NotFittedError('The {} is not fitted yet.'.format(self.__class__))
        return self.model.predict_proba(X[self.model_cols])


class StepwiseLogisticRegression(StepwiseBase):
    """ Stepwise Logistic Regression
        Reference: https://newonlinecourses.science.psu.edu/stat501/node/329/
    """

    def __init__(self, 
                cols=None, 
                alpha_enter=0.15, 
                alpha_exit=0.15,
                criteria='p-value',
                max_feature=None,
                mode='forward',
                method='bfgs', 
                refit=True, 
                **kwargs):
        """
        :param alpha_enter: alpha value below which a new feature will be added to the model
        :param alpha_exit: alpha value above which a feature will be removed from the model
        :param criteria: The criteria for picking the entry feature, currently supporting 
                ['p-value', 'aic', 'wald-chi2']
        :param mode: 'forward', 'backward' or 'bidirectional'
        :param method: Optimizer used to fit logistic model
        :param max_feature: The maximum number of features in the final model
        :param kwargs: keyword arguments for fitting the LogisticRegression model in fit() method
        """
        super().__init__(cols, 
                        max_feature=max_feature, 
                        mode=mode, 
                        method=method, 
                        refit=refit,
                        **kwargs)
        self.alpha_enter = alpha_enter
        self.alpha_exit = alpha_exit
        if criteria not in ('p-value', 'aic', 'wald-chi2'):
            raise ValueError('The supported criterias are: {}'.format(['p-value', 'aic', 'wald-chi2']))
        self.criteria = criteria

    def add_feature(self, fit_results):
        fet_rank = list()
        for cand, logit_result in fit_results:
            p_value = logit_result.pvalues[cand]
            if p_value > self.alpha_enter:
                continue

            if self.criteria == 'p-value':
                fet_rank.append((p_value, cand))
            elif self.criteria == 'aic':
                fet_rank.append((logit_result.aic, cand))
            else:
                wald = logit_result.params / np.square(logit_result.bse)
                # the candidate feature is always the last one
                wald = list(wald)[-1]
                # use the negative wald-chi2
                fet_rank.append((-wald, cand))
            
        if fet_rank:
            return sorted(fet_rank)[0][1]
        else:
            return None

    def drop_features(self, logit_result):
        p_values = logit_result.pvalues
        drop_cols = p_values[p_values > self.alpha_exit].index.tolist()
        return drop_cols


class IncrementalLogisticRegression(StepwiseLogisticRegression):
    """ Adding one feature at a time, the order of addition is predefined """

    def __init__(self, 
            cols=None, 
            sort_method='tree',
            alpha_enter=0.15, 
            alpha_exit=0.15,
            criteria='p-value',
            max_feature=None,
            method='bfgs', 
            refit=True, 
            **kwargs):
        super().__init__(self, 
                alpha_enter=alpha_enter, 
                alpha_exit=alpha_exit,
                criteria=criteria,
                max_feature=max_feature,
                mode='forward',
                method=method, 
                refit=refit, 
                **kwargs)
        self.cols = cols
        self.sort_method = sort_method

    def fit(self, X, y, **fit_params):
        cols = self.cols or X.columns.tolist()
        
        if self.sort_method is None:
            super().fit(X, y, **fit_params)
        elif self.sort_method == 'tree':
            cols = sort_columns_tree(X, y, cols=cols)
            super().fit(X[cols], y, **fit_params)
        elif self.sort_method in ('chi2', 'logistic'):
            cols = sort_columns_logistic(X, y, cols=cols)
            super().fit(X[cols], y, **fit_params)
        else:
            raise ValueError('Only {} is supported for sort method.'.format(['tree', 'logistic']))
