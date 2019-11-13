try:
	import scikitplot as skplt
except ImportError:
	raise ValueError('Plotting uses scikit-plot internally. '
					  'Install scikit-plot with `pip install scikit-plot`')


from scikitplot.metrics import (plot_confusion_matrix, plot_roc, plot_ks_statistic, plot_precision_recall,
								plot_cumulative_gain, plot_lift_curve)


