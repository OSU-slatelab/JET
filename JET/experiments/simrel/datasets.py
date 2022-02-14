'''
Wrappers for medical similarity/relatedness datasets
'''

__all__ = ['UMNSRS_Relatedness', 'UMNSRS_Similarity', 'WikiSRS_Relatedness', 'WikiSRS_Similarity']

import codecs
import embedding_methods as em

_UMNSRS_similarity = '../data/experiments/simrel/UMNSRS/UMNSRS_similarity.csv'
_UMNSRS_relatedness = '../data/experiments/simrel/UMNSRS/UMNSRS_relatedness.csv'

_WikiSRS_relatedness = '../data/experiments/simrel/WikiSRS/WikiSRS_relatedness.csv'
_WikiSRS_similarity = '../data/experiments/simrel/WikiSRS/WikiSRS_similarity.csv'

class BaseEvalSet:
    data = []

    def __init__(self, fname, lineaction, repr_method):
        self.data = []
        self.full_data = []
        self.name = ""
        if repr_method in [em.WORD, em.TERM]:
            use_strings = True
        else: use_strings = False

        hook = codecs.open(fname, 'r', 'utf-8')
        hook.readline() # skip headers
        for line in hook:
            lineaction(line.strip(), use_strings)
        hook.close()

class UMNSRS_Relatedness(BaseEvalSet):
    """UMNSRS relatedness dataset.

    588 concept pairs, scored 1-1600 (1=least similar, 1600=most similar).

    self.data instances are in form (CUI1, CUI2, Mean, StdDev)
    """
    def __init__(self, repr_method):
        def action(line, use_strings):
            (mn, stddev, term1, term2, cui1, cui2) = chunks = line.split(',')
            (term1, term2, cui1, cui2) = [s[1:-1] for s in chunks[2:]]
            if not use_strings: self.data.append((cui1, cui2, float(mn)))
            else: self.data.append((term1, term2, float(mn)))
            self.full_data.append((cui1, term1, cui2, term2, float(mn)))
        super(UMNSRS_Relatedness, self).__init__(_UMNSRS_relatedness, action, repr_method)
        self.name = "UMNSRS_Relatedness"

class UMNSRS_Similarity(BaseEvalSet):
    """UMNSRS similarity dataset.

    566 concept pairs, scored 1-1600 (1=least similar, 1600=most similar).

    self.data instances are in form (CUI1, CUI2, Mean, StdDev)
    """
    def __init__(self, repr_method):
        def action(line, use_strings):
            (mn, stddev, term1, term2, cui1, cui2) = chunks = line.split(',')
            (term1, term2, cui1, cui2) = [s[1:-1] for s in chunks[2:]]
            if not use_strings: self.data.append((cui1, cui2, float(mn)))
            else: self.data.append((term1, term2, float(mn)))
            self.full_data.append((cui1, term1, cui2, term2, float(mn)))
        super(UMNSRS_Similarity, self).__init__(_UMNSRS_similarity, action, repr_method)
        self.name = "UMNSRS_Similarity"

class WikiSRS_Similarity(BaseEvalSet):
    """Encyclopedia similarity dataset.

    688 concept pairs, scored 0-100 (0=least similar, 100=most similar).

    self.data instances are in form (CUI1, CUI2, Mean)
    """
    def __init__(self, repr_method):
        def action(line, use_strings):
            (cui1, cui2, lbl1, lbl2, mn, stddev, scores) = chunks = line.split('\t')
            if not use_strings: self.data.append((cui1, cui2, float(mn)))
            else: self.data.append((lbl1, lbl2, float(mn)))
            self.full_data.append((cui1, lbl1, cui2, lbl2, float(mn)))
        super(WikiSRS_Similarity, self).__init__(_WikiSRS_similarity, action, repr_method)
        self.name = 'WikiSRS_Similarity'

class WikiSRS_Relatedness(BaseEvalSet):
    """Encyclopedia relatedness dataset.

    688 concept pairs, scored 0-100 (0=least related, 100=most related).

    self.data instances are in form (CUI1, CUI2, Mean)
    """
    def __init__(self, repr_method):
        def action(line, use_strings):
            (cui1, cui2, lbl1, lbl2, mn, stddev, scores) = chunks = line.split('\t')
            if not use_strings: self.data.append((cui1, cui2, float(mn)))
            else: self.data.append((lbl1, lbl2, float(mn)))
            self.full_data.append((cui1, lbl1, cui2, lbl2, float(mn)))
        super(WikiSRS_Relatedness, self).__init__(_WikiSRS_relatedness, action, repr_method)
        self.name = 'WikiSRS_Relatedness'
