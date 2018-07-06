'''
Implements some Information Retrieval relevant metrics.
'''

__all__= ['AveragePrecision', 'ReciprocalRank', 'AP_RR', 'MeanReciprocalRank', 'DCG', 'NDCG']

import numpy as np

def DiscountedCumulativeGain(ranked_labels, k):
    '''Calculates the discounted cumulative gain of a ranking
    with respect to the true relevance labels.

    Parameters:
        ranked_labels :: the true relevance labels of items, in their
                         predicted rank order
        k             :: the position limit (DGC@k)
    '''
    dcg = 0.
    for p in range(k):
        if p >= len(ranked_labels): break
        dcg += (
            #((2**ranked_labels[p]) - 1) /
            float(ranked_labels[p]) /
            np.log2(p+2)                   # p indexed from 0
        )
    return dcg
# alias
DCG = DiscountedCumulativeGain

def NormalizedDiscountedCumulativeGain(ranked_labels, true_ranking, k):
    '''Calculates the normalized DCG up to position p of a predicted
    ranking with respect to the true ranking by relevance label.

    Parameters:
        ranked_labels :: the true relevance labels of items, in their
                         predicted rank order
        true_ranking  :: the true rank order of relevance labels
        k             :: the position limit (DGC@k)
    '''
    dcg = DCG(ranked_labels, k)
    ideal_dcg = DCG(true_ranking, k)
    return dcg/ideal_dcg
# alias
NDCG = NormalizedDiscountedCumulativeGain

def AveragePrecision(truth, ranked):
    '''Calculates the average precision of a ranked sequence,
    given the set of "true" elements.

    Parameters
        truth  :: the set of "true" elements to find in the ranked list
        ranked :: the ranked sequence of elements to search
    '''
    (ap, _) = _AP_RR(truth, ranked)
    return ap

def ReciprocalRank(truth, ranked):
    '''Calculates the reciprocal rank of one or more "true"
    elements in a ranked sequence of elements.

    If len(truth) > 1, the maximum reciprocal rank for any
    elements of truth will be returned.

    Parameters
        truth  :: one or more "true" elements to find in the ranked list
        ranked :: the ranked sequence of elements to search
    '''
    (_, rr) = _AP_RR(truth, ranked, rr_only=True)
    return rr

def MeanReciprocalRank(result_pairs):
    '''Calculates the Mean Reciprocal Rank for a list of query
    results.

    Each element in the input list should be of the form
        ( <true element(s) to search for>, <ranked results> )
    '''
    rrs = []
    for (truth, ranked) in result_pairs:
        rrs.append(ReciprocalRank(truth, ranked))
    return np.mean(rrs)

def AP_RR(truth, ranked):
    '''Calculates both average precision and reciprocal rank
    for a set of "true" elements in a ranked sequence.

    Returns (avg_precision, reciprocal_rank)
    '''
    return _AP_RR(truth, ranked, rr_only=False)

def _AP_RR(truth, ranked, rr_only=False):
    if not type(truth) in [set,list,tuple]:
        truth = set([truth])
    else:
        truth = set(truth)

    cur_ix, num_found = 0, 0
    ap_summer, rr = 0, -1
    while num_found < len(truth) and cur_ix < len(ranked):
        if ranked[cur_ix] in truth:
            num_found += 1
            in_truth = True
        else:
            in_truth = False

        cur_ix += 1

        # save reciprocal rank (first correct index)
        if in_truth and rr == -1:
            rr = 1./cur_ix
            if rr_only: break  # can quit here

        # if a "true" value, get precision at this recall level
        if in_truth:
            ap_summer += num_found / cur_ix

        #print('  %10s  %12s  Precision: %.4f' % (
        #    ranked[cur_ix-1],
        #    ('Found!' if (ranked[cur_ix-1] in truth) else ''),
        #    ap_summer / cur_ix
        #))
    ap = ap_summer / len(truth)

    return (ap, rr)

def _testmetrics():
    '''Tests IR metrics. calculation with data taken from
    Tests Average Precision with data taken (some modification) from
        http://fastml.com/what-you-wanted-to-know-about-mean-average-precision/
    Tests Reciprocal Rank calculation with data taken from
        https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    '''

    header = lambda msg: print('{0}\n{1}\n{0}'.format('-'*30, msg))

    header('Testing ReciprocalRank and MRR')
    sets = (
        ('cats', ['catten', 'cati', 'cats'], 1./3),
        ('tori', ['torii', 'tori', 'toruses'], 1./2),
        ('viruses', ['viruses', 'virii', 'viri'], 1./1)
    )
    for (truth, ranked, expected) in sets:
        print('Expected: %.4f   Actual: %.4f' % (expected, ReciprocalRank(truth, ranked)))
    print('Expected MRR: %.4f   Actual: %.4f' % (11./18, MeanReciprocalRank([s[:2] for s in sets])))

    header('Testing AveragePrecision')
    truth = set([1,2,3,4,5])    # 5 true elements to find
    ranked = [6,4,7,1,2]        # only contains 3 of the 5 elements
    print('Expected: 0.32   Actual: %.4f' % AveragePrecision(truth, ranked))

    header('Testing DCG')
    truth = [3,3,3,2,2,2,1,0]
    ranked = [3,2,3,0,1,2]
    print('Expected: 6.861   Actual: %.4f' % DCG(ranked, 6))
    print('Expected: 0.785   Actual: %.4f' % NDCG(ranked, truth, 6))

if __name__ == '__main__':
    _testmetrics()
