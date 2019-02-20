def ks_score(y_true, y_score):
    """ Calculating the Kolmogorov-Smirnov score"""
    from sklearn import metrics

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    return max(tpr - fpr)


def lift_table(prediction, label, bins=10, mode='equal_width'):
    """ Create lift table given cutoff point or number of bins
    """
    import pandas as pd

    lift_table = pd.DataFrame({'prediction': prediction, 'label': label})

    if isinstance(bins, list):
        lift_table['score_range'] = pd.cut(lift_table['prediction'], bins)
    else:
        if mode == 'equal_width':
            lift_table['score_range'] = pd.cut(lift_table['prediction'], bins)
        elif mode.startswith('equal_freq'):
            n_perbin = len(lift_table) // bins
            lift_table = lift_table.sort_values('prediction', ascending=False) \
                .reset_index(drop=True).reset_index()
            lift_table['score_range'] = (lift_table['index'] / n_perbin).astype(int)
        else:
            raise ValueError('Mode not supported')

    lift_table = lift_table.groupby('score_range')['label'].agg({'n_sample': 'count', 'n_pos': 'sum'}).reset_index()
    lift_table['n_neg'] = lift_table['n_sample'] - lift_table['n_pos']
    lift_table['pct_pos'] = lift_table['n_pos'] / lift_table['n_sample']
    lift_table = lift_table.sort_values('score_range', ascending=False)
    lift_table['cum_pct_pos'] = lift_table['n_pos'].cumsum() / lift_table['n_sample'].sum()
    lift_table['pct_sample'] = lift_table['n_sample'] / lift_table['n_sample'].sum()
    return lift_table

