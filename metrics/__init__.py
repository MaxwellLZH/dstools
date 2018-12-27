from sklearn import metrics


def ks_score(y_true, y_score):
    """ Calculating the Kolmogorov-Smirnov score"""
    fpr, tpr, _ = metrics.roc_auc_score(y_true, y_score)
    return max(tpr - fpr)
