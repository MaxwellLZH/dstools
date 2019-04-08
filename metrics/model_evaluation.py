import numpy as np


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


def psi(expected_array, actual_array, buckets=20, buckettype='bins'):
    """ Calculate the PSI for a single variable
    Args:
       expected_array: numpy array of original values
       actual_array: numpy array of new values, same size as expected
       buckets: number of percentile ranges to bucket the values into
       buckettype: available choices: ['bins', 'quantiles']
    Returns:
       psi_value: calculated PSI value
    """
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    if buckettype == 'bins':
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif buckettype == 'quantiles':
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

    expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
    actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

    def sub_psi(e_perc, a_perc):
        """ Calculate the actual PSI value from comparing the values.
           Update the actual value to a very small number if equal to zero """
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return value

    psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))
    return psi_value