from .base import Binning
from .supervised import ChiSquareBinning
from .unsupervised import EqualFrequencyBinning, EqualWidthBinning, equal_frequency_binning, equal_width_binning


__all__ = [
    'Binning',
    'ChiSquareBinning',
    'EqualWidthBinning',
    'EqualFrequencyBinning',
    'equal_width_binning',
    'equal_frequency_binning'
]