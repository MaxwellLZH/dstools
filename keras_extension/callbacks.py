from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import roc_auc_score


class BaseEpochEvaluator(Callback):
    """ Base class for performing evaluation after every epoch."""
    eval_name = ''
    eval_f = None
    mode = 'min'

    def __init__(self, training_data=None, validation_data=None,
                 earlystopping=False, patience=np.inf):
        super().__init__()
        self.train = training_data
        self.val = validation_data
        self.metrics = list()

        if earlystopping and validation_data is None:
            raise ValueError('Validation data is needed to earlystopping to work.')
        self.earlystopping = earlystopping
        self.patience = patience
        self.stopped_epoch = 0
        self.wait = 0
        self.best = None

        if self.mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.mode == 'min' else -np.inf

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        train_str, train_metric = self._eval(self.train, 'train')
        val_str, val_metric = self._eval(self.val, 'validation')
        stats = {'epoch': epoch}
        if train_metric is not None:
            stats['train_{}'.format(self.eval_name)] = train_metric
        if val_metric is not None:
            stats['val_{}'.format(self.eval_name)] = val_metric
        self.metrics.append(stats)
        print('\n'.join([train_str, val_str]))

        # check for early stopping
        if self.monitor_op(val_metric, self.best):
            self.best = val_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def _eval(self, data=None, mode='train'):
        if data:
            x, y = data
            pred = self.model.predict(x)
            metric = self.__class__.eval_f(y, pred)
            return '{} on {} data: {:.2%}.'.format(self.eval_name.upper(), mode, metric), metric
        else:
            return '', None


class AUCEpochEvaluator(BaseEpochEvaluator):
    """ Callback for calculating AUC after every epoch """
    eval_name = 'auc'
    eval_f = roc_auc_score
    mode = 'max'
