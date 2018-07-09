#!/bin/bash
#
# Demo script showing the process of running JET on a new corpus
#  (1) build the token-level term mapping for annotation
#  (2) automatically annotate the corpus (with string matching)
#  (3) train JET on annotations
#

if [ ! -d data/demo ]; then
    mkdir -p data/demo
fi

# get the demo corpus
curl -ko data/demo/demo_corpus.zip https://slate.cse.ohio-state.edu/JET/data/demo_corpus.zip
cd demo_data && unzip demo_corpus.zip && cd ../

# build the term map for preprocessing
python3 -m preprocessing.readstrings \
    --ignore-stopword-terms \
    data/demo/test_entity_strings.csv \
    data/demo
# annotate the test corpus
python3 -m preprocessing.tagcorpus \
    --threads=3 \
    data/demo/test_corpus.txt \
    data/demo/test_entity_strings.csv.ngram.pkl.gz \
    data/demo/test_corpus.annotations \
    --max-lines=3

# compile JET
make bin/JET

# run it on the test annotations
bin/JET \
    -size 10 \
    -binary 0 \
    -negative 2 \
    -plaintext data/demo/test_corpus.txt \
    -annotations data/demo/test_corpus.annotations \
    -model data/demo \
    -term-map data/demo/test_entity_strings.csv.tagmap.txt \
    -string-map data/demo/test_entity_strings.csv.strmap.txt
