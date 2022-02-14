'''
Wrapper and data models for NLM-WSD dataset

Requires NLM-WSD datafiles, downloadable from
   https://wsd.nlm.nih.gov/collaboration.shtml#MSH_WSD
'''

_datadir = '.../NLM-WSD/MSHCorpus'

import os
import glob
from hedgepig_logger import log
from . import parser

class NLM_WSD:
    
    def __init__(self, datadir=None, verbose=False):
        self._ambig_sets = {}

        if not datadir: datadir = _datadir
        for f in glob.glob(os.path.join(datadir, '*_pmids_tagged.arff')):
            if verbose: log.writeln('  >> Parsing %s' % f)
            ambig_set = parser.parseFile(f)
            concept = os.path.basename(f).split('_')[0]
            self._ambig_sets[concept] = ambig_set

    def __getitem__(self, key):
        return self._ambig_sets[key]

    def __iter__(self):
        return iter(self._ambig_sets.values())

    def __len__(self):
        return len(self._ambig_sets)
