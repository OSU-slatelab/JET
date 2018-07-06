'''
Collection of scientific metrics.
'''

__all__ = [
    'SpearmansRho', 
    'CosineSimilarity', 'CosineDistance',
    'ReciprocalRank', 'AveragePrecision', 'AP_RR', 'MeanReciprocalRank'
]

from .spearmans import SpearmansRho
from .cosine import CosineSimilarity, CosineDistance
from .ir_metrics import *
