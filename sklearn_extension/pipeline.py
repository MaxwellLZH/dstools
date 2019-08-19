from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
from sklearn.base import clone

__all__ = ['Pipeline']


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

    # If the result doesn't have y, we pass through the original y label
    if isinstance(res, (list, tuple)):
        assert len(res) == 2, ('If the transformed result contains both X and y, '
                                'it should only contain two elements')
        X, y = res
    else:
        X = res

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
                return last_step.fit_transform(Xt, yt, **fit_params)
            else:
                return last_step.fit(Xt, yt, **fit_params).transform(Xt)