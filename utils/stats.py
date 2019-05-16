import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def plot_distribution(rv, rv_options=None, bins=100, ax=None):
    """ Show the plot for the specified distribution
    :param rv: A random variable from scipy.stats or the name of the distribution
    :param rv_options: A dictionary of options for the random variable, such as degree of freedom for chi2.
    """
    if isinstance(rv, str):
        err_msg = ('{0} is not included in scipy.stats. '
                   'Find all the available distributions at https://docs.scipy.org/doc/scipy/reference/stats.html').format(
            rv)
        try:
            rv = getattr(stats, rv)
        except:
            raise ValueError(err_msg)

    rv_options = rv_options or dict()
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(rv, stats.rv_continuous):
        # continuous variable
        x = np.linspace(rv.ppf(0.01, **rv_options), rv.ppf(0.99, **rv_options), bins)
        y = rv.pdf(x, **rv_options)
        ax.plot(x, y, 'r-', lw=3)
    else:
        # discrete variable
        x = np.arange(rv.ppf(0.01, **rv_options), rv.ppf(0.99, **rv_options))
        y = rv.pmf(x, **rv_options)
        ax.plot(x, y, 'bo', ms=8)
        ax.vlines(x, 0, y, colors='r', lw=3, alpha=0.5)

    plt.show()
    return fig