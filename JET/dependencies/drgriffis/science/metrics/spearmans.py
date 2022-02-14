'''
Implements Spearman's Rho calculation.
'''

__all__= ['SpearmansRho']

import numpy as np
from drgriffis.common import util

def SpearmansRho(valpairs):
    '''Calculates Spearman's rank correlation coefficient (Spearman's rho)
    between two metrics over the same dataset.

    Data should be input as (metric1, metric2) pairs, where metric1 and
    metric2 are two metrics on the same datapoint.
    '''
    rank_pairs = valsToRanks(valpairs)

    n = len(valpairs)
    sq_diffs = np.array([np.square(r2-r1) for (r1,r2) in rank_pairs])

    rho = 1 - (
        ( 6 * np.sum(sq_diffs) ) /
        ( n * ( (n**2) - 1 ) )
    )
    return rho

class ItemPair:
    item1 = None
    item2 = None

def valsToRanks(valpairs):
    '''Converts pairs of values to pairs of ranks.

    Each (val1, val2) pair is converted to (rank1, rank2),
    where rank1 is the position of val1 within all val1s,
    and rank2 is the position of val2 within all val2s.
    '''
    ranks1 = toranks([vp[0] for vp in valpairs])
    ranks2 = toranks([vp[1] for vp in valpairs])
    return [
        (ranks1[i], ranks2[i])
            for i in range(len(valpairs))
    ]

def toranks(vals):
    sorts = np.sort(np.array(vals)).tolist()
    ranks = np.array([sorts.index(v) for v in vals]) + 1
    return ranks

def _testrho():
    '''Tests Spearman's Rho calculation with data taken from
    Wikipedia page:
        https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    '''
    vals1 = [86, 97, 99, 100, 101, 103, 106, 110, 112, 113]
    vals2 = [0, 20, 28, 27, 50, 29, 7, 17, 6, 12]
    valpairs = [(vals1[i], vals2[i]) for i in range(len(vals1))]
    rho = SpearmansRho(valpairs)

    print('Expected: -0.1758   Actual: %.4f' % rho)
if __name__ == '__main__':
    _testrho()
