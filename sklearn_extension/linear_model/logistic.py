from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
import statsmodels.api as sm
import numpy as np
import pandas as pd


class StepwiseLogisticRegression(BaseEstimator, ClassifierMixin):
    """ Stepwise Logistic Regression
        Reference: https://newonlinecourses.science.psu.edu/stat501/node/329/
    """
    
    def __init__(self, cols=None, alpha_enter=0.15, alpha_exit=0.15, mode='forward', **kwargs):
        """
        :param cols: columns ranked by importance from high to low, ex. iv
        :param alpha_enter: alpha value below which a new feature will be added to the model
        :param alpha_exit: alpha value above which a feature will be removed from the model
        :param mode: 'forward', 'backward' or 'bidirectional'
        :param kwargs: keyword arguments for fitting the LogisticRegression model in 
            fit() method
        """
        if mode not in ('forward', 'backward', 'bidirectional'):
            raise ValueError('Only support forward, backward and bidirectional mode.')
        self.cols = cols
        self.alpha_enter = alpha_enter
        self.alpha_exit = alpha_exit
        self.mode = mode
        self.fit_kwargs = kwargs
        
        self.model_cols = None
        self.model = None
        
    def forward_step(self, X, y, candidate_cols, model_cols):
        """ Perform a single forward step
            Return updated candidate columns, model columns and a boolean indicating
            whether it should be the last step
        """
        fet_p_value = list()
        for cand in candidate_cols:
            logit_model = sm.Logit(y, X[model_cols+[cand]])
            try:
                logit_result = logit_model.fit(disp=False, method='lbfgs')
            except np.linalg.LinAlgError:
                """ usually caused by multi-colinearity, simply ignore the column """
                continue
                
            p_value = logit_result.pvalues[cand]
            if p_value <= self.alpha_enter:
                fet_p_value.append((logit_result.pvalues[cand], cand))
        
        if fet_p_value:
            # the next added feature is the one with the smallest p value
            added_col = sorted(fet_p_value)[0][1]
            candidate_cols = [c for c in candidate_cols if c != added_col]
            
            return candidate_cols, model_cols + [added_col], len(candidate_cols)==0
        else:
            return candidate_cols, model_cols, True
        
    def backward_step(self, X, y, model_cols):
        """
        Return model coumns and a boolean indicating whether it should be the last step
        """
        model = sm.Logit(y, X[model_cols]).fit(disp=False, method='lbfgs')
        p_values = model.pvalues
        drop_cols = p_values[p_values>=self.alpha_exit].index.tolist()
        model_cols = [c for c in model_cols if c not in drop_cols]
        # stop the stepping process when there's no more column to drop 
        # or there's only one column in the model (the constant)
        is_last_step = len(drop_cols)==0 or len(model_cols)==1
        return model_cols, is_last_step
            
    def fit(self, X, y, **fit_params):
        self.cols = cols = self.cols or X.columns.tolist()
        X = sm.add_constant(X.copy())
        
        need_forward_step = self.mode in ('forward', 'bidirectional')
        need_backward_step = self.mode in ('bidirectional', 'backward')
        
        if need_forward_step:
            candidate_cols, model_cols = self.cols, ['const']
        else:
            candidate_cols, model_cols = None, self.cols + ['const']
        
        stop_sign = False
        # ignore the convergence warnings for now
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            while not stop_sign:
                if need_forward_step:
                    candidate_cols, model_cols, forward_stop_sign = \
                            self.forward_step(X, y, candidate_cols, model_cols)
                if need_backward_step:
                    model_cols, backward_stop_sign = self.backward_step(X, y, model_cols)

                if self.mode=='forward':
                    stop_sign = forward_stop_sign
                elif self.mode == 'bidirectional':
                    stop_sign = forward_stop_sign and backward_stop_sign
                else:
                    stop_sign = backward_stop_sign
        
        model_cols.remove('const')
        self.model_cols = model_cols
        
        from sklearn.linear_model import LogisticRegression
        
        if len(self.model_cols) == 0:
            raise ValueError('No column was selected.')
        LR = LogisticRegression(**self.fit_kwargs)
        LR.fit(X[model_cols], y)
        self.model = LR
        return self
    
    def predict(self, X):
        if self.model_cols is None:
            raise NotFittedError('The StepwiseLogisticRegression is not fitted yet.')
        
        return self.model.predict(X[self.model_cols])
    
    def predict_proba(self, X):
        if self.model_cols is None:
            raise NotFittedError('The StepwiseLogisticRegression is not fitted yet.')
        
        return self.model.predict_proba(X[self.model_cols])


class IncrementalLogisticRegression(BaseEstimator, ClassifierMixin):
    """ Incremental Logistic Regression """
    
    def __init__(self, cols=None, sort=False, max_feature=None, alpha_enter=0.15, **kwargs):
        """
        :param cols: columns ranked by importance from high to low, ex. iv
        :param max_feature: maximum number of features to use, default using all the features
        :param sort: sort the features according to wald chi2
        :param alpha_enter: alpha value below which a new feature will be added to the model
        :param kwargs: keyword arguments for fitting the LogisticRegression model in 
            transform() method
        """
        self.cols = cols
        self.sort = (cols is None) or sort
        self.max_feature = max_feature or np.inf
        self.alpha_enter = alpha_enter
        self.fit_kwargs = kwargs
        
        self.model_cols = None
        self.model = None
        
    def sort_columns_logistic(self, X, y, cols):
        """ Sort columns according to wald_chi2 """
        logit_result = sm.Logit(y, X[cols+['const']]).fit()
        wald_chi2 = np.square((logit_result.params) / np.square(logit_result.bse))
        wald_chi2 = pd.DataFrame({'chi2': wald_chi2, 'feature': cols})
        sorted_cols = wald_chi2.sort_values('chi2', ascending=False).feature.tolist()
        sorted_cols.remove('const')
        return sorted_cols
    
    def sort_columns_tree(self, X, y, cols):
        """ Sort columns according to feature importance in tree method"""
        from sklearn.ensemble import RandomForestClassifier
        
        RF = RandomForestClassifier()
        RF.fit(X[cols], y)
        importance = pd.DataFrame({'importance': RF.feature_importances_, 'feature': cols})
        sorted_cols = importance.sort_values('importance', ascending=False).feature.tolist()
        return sorted_cols
        
    def forward_step(self, X, y, candidate_col, model_cols):
        """ Perform a single forward step
            Return updated candidate columns, model columns and a boolean indicating
            whether it should be the last step
        """
        logit_model = sm.Logit(y, X[model_cols+[candidate_col]])
        try:
            logit_result = logit_model.fit(disp=False, method='lbfgs')
        except np.linalg.LinAlgError:
            """ usually caused by multi-colinearity, simply ignore the column """
            return model_cols
              
        p_value = logit_result.pvalues[candidate_col]
        if p_value <= self.alpha_enter:
            return model_cols + [candidate_col]
        else:
            return model_cols
                
    def fit(self, X, y, **fit_params):
        self.cols = cols = self.cols or X.columns.tolist()    
        
        if self.sort:
            self.cols = cols = self.sort_columns_tree(X, y, cols)
        
        X = sm.add_constant(X.copy(), has_constant='add')

        # ignore the convergence warnings for now
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            model_cols = ['const']
            for col in cols:
                model_cols = self.forward_step(X, y, col, model_cols)
                # including a constant column thus max_feature + 1
                if len(model_cols) >= self.max_feature + 1:
                    print('Early stopped at {} features.'.format(self.max_feature))
                    break
                    
        model_cols.remove('const')
        self.model_cols = model_cols
        
        from sklearn.linear_model import LogisticRegression
        
        if len(self.model_cols) == 0:
            raise ValueError('No column was selected.')
        LR = LogisticRegression(**self.fit_kwargs)
        LR.fit(X[model_cols], y)
        self.model = LR
        return self
    
    def predict(self, X):
        if self.model_cols is None:
            raise NotFittedError('The IncrementalLogisticRegression is not fitted yet.')
        
        return self.model.predict(X[self.model_cols])
    
    def predict_proba(self, X):
        if self.model_cols is None:
            raise NotFittedError('The IncrementalLogisticRegression is not fitted yet.')
        
        return self.model.predict_proba(X[self.model_cols])