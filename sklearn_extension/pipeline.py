from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
from sklearn.base import clone, BaseEstimator, TransformerMixin
import pandas as pd
import inspect


__all__ = ['Pipeline', 'Inspect']


def _inspect_step(X, y=None):
    """ A step that can be plugged into the pipeline to inspect the 
        intermediary output within the pipeline """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    print('=' * 80)
    print('Shape of X: {}'.format(X.shape))
    print('=' * 80)
    print('Missing percentage: \n{}'.format(X.isnull().mean()))
    print('=' * 80)
    if y is not None:
        print('Label distribution: \n{}'.format(pd.Series(y).value_counts(True)))
        print('=' * 80)
    return X, y


class Inspect(BaseEstimator, TransformerMixin):
    """ A step that can be plugged into the pipeline to inspect the 
        intermediary output within the pipeline """

    def fit(self, X, y=None, **fit_params):
        _inspect_step(X, y)
        return self

    def transform(self, X):
        return X



def pass_y(obj):
    """ Determine if we should pass y into the transform method """
    signature = inspect.signature(getattr(obj, 'transform'))
    return 'y' in signature.parameters


def _wrap_result(res, y):
    # If the result doesn't have y, we pass through the original y label
    if isinstance(res, (list, tuple)):
        assert len(res) == 2, ('If the transformed result contains both X and y, '
                                'it should only contain two elements')
        X, y = res
    else:
        X = res
    return X, y


def _fit_transform_one(transformer,
                       X,
                       y,
                       weight,
                       message_clsname='',
                       message=None,
                       **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, 'fit_transform'):
            res = transformer.fit_transform(X, y, **fit_params)
        else:
            res = transformer.fit(X, y, **fit_params).transform(X)

    X, y = _wrap_result(res, y)

    if weight is None:
        return X, y, transformer
    return X * weight, y, transformer



class Pipeline(_Pipeline):
    """ A dropin replacement for Scikit-learn Pipeline object that supports 
        transforming X and y at the same time. Most used for performing over/under sampling, 
        resampling etc during the transformation process
    """
    def _fit(self, X, y=None, **fit_params):
        """ Overwrite the _fit method """  
        self.steps = list(self.steps)
        self._validate_steps()
        
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}
        for pname, pval in fit_params.items():
            if '__' not in pname:
                raise ValueError(
                    "Pipeline.fit does not accept the {} parameter. "
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipeline.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`.".format(pname))
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if (transformer is None or transformer == 'passthrough'):
                with _print_elapsed_time('Pipeline',
                                         self._log_message(step_idx)):
                    continue

            if hasattr(memory, 'location'):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, 'cachedir'):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)
            
            Xt, yt, fitted_transformer = fit_transform_one_cached(
                cloned_transformer, Xt, y, None,
                message_clsname='Pipeline',
                message=self._log_message(step_idx),
                **fit_params_steps[name])
            
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator == 'passthrough':
            return Xt, yt, {}
        return Xt, yt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
            """Fit the model
            Fit all the transforms one after the other and transform the
            data, then fit the transformed data using the final estimator.
            Parameters
            ----------
            X : iterable
                Training data. Must fulfill input requirements of first step of the
                pipeline.
            y : iterable, default=None
                Training targets. Must fulfill label requirements for all steps of
                the pipeline.
            **fit_params : dict of string -> object
                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.
            Returns
            -------
            self : Pipeline
                This estimator
            """
            Xt, yt, fit_params = self._fit(X, y, **fit_params)
            with _print_elapsed_time('Pipeline',
                                     self._log_message(len(self.steps) - 1)):
                if self._final_estimator != 'passthrough':
                    self._final_estimator.fit(Xt, yt, **fit_params)
            return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator
        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples
        """
        last_step = self._final_estimator
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if last_step == 'passthrough':
                return Xt, yt
            if hasattr(last_step, 'fit_transform'):
                return _wrap_result(last_step.fit_transform(Xt, yt, **fit_params), yt)
            else:    
                return _wrap_result(last_step.fit(Xt, yt, **fit_params).transform(Xt), yt)



    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.
        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        y_pred : array-like
        """
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][-1].fit_predict(Xt, yt, **fit_params)
        return y_pred


    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **predict_params):
        """Apply transforms to the data, and predict with the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.
        Returns
        -------
        y_pred : array-like
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            # any step that modifies label should be skipped during the predict process
            if pass_y(transform):
                continue
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt, **predict_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        """Apply transforms, and predict_proba of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            # any step that modifies label should be skipped during the predict process
            if pass_y(transform):
                continue
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):
        """Apply transforms, and decision_function of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            # any step that modifies label should be skipped during the predict process
            if pass_y(transform):
                continue
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].decision_function(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X):
        """Apply transforms, and predict_log_proba of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            # any step that modifies label should be skipped during the predict process
            if pass_y(transform):
                continue
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_log_proba(Xt)

    @property
    def transform(self):
        """Apply transforms, and transform with the final estimator
        This also works where final estimator is ``None``: all prior
        transformations are applied.
        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
        """
        # _final_estimator is None or has transform, otherwise attribute error
        # XXX: Handling the None case means we can't use if_delegate_has_method
        if self._final_estimator != 'passthrough':
            self._final_estimator.transform
        return self._transform

    def _transform(self, X, y=None):
        Xt, yt = X, y
        for _, _, transform in self._iter():
            if pass_y(transform):
                Xt, yt = transform.transform(Xt, yt)
            else:
                Xt = transform.transform(Xt)
        return Xt, yt

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, sample_weight=None):
        """Apply transforms, and score with the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.
        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.
        Returns
        -------
        score : float
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            # any step that modifies label should be skipped during the predict process
            if pass_y(transform):
                continue
            Xt = transform.transform(Xt)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, y, **score_params)


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    ppl = Pipeline([
            ('ss', StandardScaler()),
            ('rb', RobustScaler()),
            ('ins', Inspect()),
            ('lr', LogisticRegression())
        ])

    X, y = load_breast_cancer(True)
    ppl.fit(X, y)

    pred = ppl.predict_proba(X)[:, 1]
    print(roc_auc_score(y, pred))

