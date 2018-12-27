from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


class BaseEpochEvaluator(Callback):
    """ Base class for performing evaluation after every epoch."""
    eval_name = ''
    eval_f = None

    def __init__(self, training_data=None, validation_data=None):
        super().__init__()
        self.train = training_data
        self.val = validation_data
        self.metrics = list()

    def on_train_begin(self, logs=None):
        pass

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
        print(' '.join([train_str, val_str]))

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

