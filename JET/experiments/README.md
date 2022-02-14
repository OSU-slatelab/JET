## Experimental implementations

This directory contains Python-based implementations of the experiments run in the paper.  The `replicate_*.sh`
scripts will download the necessary data and run code to replicate the experimental results presented in the paper.

Implementations require:

- Tensorflow
- NumPy/SciPy
- NLTK
- [pyemblib](https://pypi.org/project/pyemblib/)
- [hedgepig-logger](https://pypi.org/project/hedgepig-logger/)

## Note on entity linking (named entity disambiguation)

For convenience, the replication script will download our pre-extracted mention files for re-use,
as the mention extraction process is not straightforward.

However, we have included our mention extraction code.  For both NLM-WSD and AIDA, the `extract_mentions.py`
script is used, which in turn calls scripts in `datasets/nlm_wsd` and `datasets/aida`.

To extract mentions, you will need to download the following data files:

- **NLM-WSD**
    + [ARFF format mention files (require UTS license)](https://wsd.nlm.nih.gov/collaboration.shtml#MSH_WSD)
- **AIDA**
    + [CoNLL 2003 NER dataset](https://www.clips.uantwerpen.be/conll2003/ner/)
    + [AIDA CoNLL-YAGO dataset](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/)
    + [PPRforNED AIDA candidates, developed by Maria Pershina](https://github.com/masha-p/PPRforNED)
