#!/bin/bash
#
# Demo script showing the process of running JET on a new corpus
#  (1) build the token-level term mapping for annotation
#  (2) automatically annotate the corpus (with string matching)
#  (3) train JET on annotations
#

# build the term map for preprocessing
python3 -m JET.preprocessing.compile_terminology \
    --ignore-stopword-terms \
    -i data/demo/test_entity_strings.csv \
    -o data/demo
# annotate the test corpus
python3 -m JET.preprocessing.tagcorpus \
    --threads=3 \
    -i data/demo/test_corpus.txt \
    -t data/demo/test_entity_strings.ngram_term_map.pkl.gz \
    -o data/demo/test_corpus.annotations \
    --max-lines=3

# compile JET
make JET/implementation/bin/JET

# run it on the test annotations
JET/implementation/bin/JET \
    -size 10 \
    -binary 0 \
    -negative 2 \
    -plaintext data/demo/test_corpus.txt \
    -annotations data/demo/test_corpus.annotations \
    -model data/demo \
    -term-map data/demo/test_entity_strings.term_to_entity_map.txt \
    -string-map data/demo/test_entity_strings.term_to_string_map.txt
