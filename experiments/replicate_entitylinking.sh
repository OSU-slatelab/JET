#!/bin/bash

PY=python3
PYTHONPATH=$(pwd)/lib

source fetch_data.sh

DATA=../data/experiments/entitylinking

#############################################################
## NLM-WSD - PubMed embeddings
#############################################################

if [ -e ${NLM_WSD} ]; then
    echo
    echo "-- Results on NLM-WSD with PubMed embeddings --"
    echo

    echo "JET: Entities + entity definitions..."
    LOGF=${DATA}/NLM-WSD/results/pubmed_entities_definitions.log
    if [ ! -e ${LOGF} ]; then
        ${PY} -m entitylinking.experiment \
            ${NLM_WSD} \
            --entities=../data/embeddings/pubmed/entities.bin \
            --words=../data/embeddings/pubmed/words.bin \
            --ctxs=../data/embeddings/pubmed/words.bin \
            --entity-definitions=${NLM_WSD_DEFN} \
            --minibatch-size=100 \
            --combo-method=sum \
            --strings=${UMLS_2017AB_PREFSTRS} \
            --predictions=${LOGF}.predictions \
            -l ${LOGF} \
            1> /dev/null
    fi
    tail -n 5 ${LOGF}.predictions

else
    echo
    echo "Skipping NLM-WSD experiments; need to download dataset"
    echo "File can be downloaded from this page:"
    echo "   http://slate.cse.ohio-state.edu/UTSAuthenticatedDownloader/index.html?dataset=NLM_WSD_JET"
fi

#############################################################
## AIDA - Wikipedia embeddings
#############################################################

echo
echo "-- Results on AIDA with Wikipedia embeddings --"
echo

echo "JET: Entities..."
LOGF=${DATA}/AIDA/results/wikipedia_entities.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m entitylinking.experiment \
        ${AIDA} \
        --entities=../data/embeddings/wikipedia/entities.bin \
        --words=../data/embeddings/wikipedia/words.bin \
        --ctxs=../data/embeddings/wikipedia/words.bin \
        --minibatch-size=100 \
        --combo-method=sum \
        --predictions=${LOGF}.predictions \
        -l ${LOGF} \
        1> /dev/null
fi
tail -n 1 ${LOGF}.predictions

echo "JET: Entities + mention text..."
LOGF=${DATA}/AIDA/results/wikipedia_entities_mentions.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m entitylinking.experiment \
        ${AIDA} \
        --entities=../data/embeddings/wikipedia/entities.bin \
        --words=../data/embeddings/wikipedia/words.bin \
        --ctxs=../data/embeddings/wikipedia/words.bin \
        --minibatch-size=100 \
        --combo-method=sum \
        --predictions=${LOGF}.predictions \
        --use-mentions \
        -l ${LOGF} \
        1> /dev/null
fi
tail -n 1 ${LOGF}.predictions

#############################################################
## AIDA - Gigaword embeddings
#############################################################

echo
echo "-- Results on AIDA with Gigaword embeddings --"
echo

echo "JET: Entities..."
LOGF=${DATA}/AIDA/results/gigaword_entities.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m entitylinking.experiment \
        ${AIDA} \
        --entities=../data/embeddings/gigaword/entities.bin \
        --words=../data/embeddings/gigaword/words.bin \
        --ctxs=../data/embeddings/gigaword/words.bin \
        --minibatch-size=100 \
        --combo-method=sum \
        --predictions=${LOGF}.predictions \
        -l ${LOGF} \
        1> /dev/null
fi
tail -n 1 ${LOGF}.predictions
exit

echo "JET: Entities + mention text..."
LOGF=${DATA}/AIDA/results/gigaword_entities_mentions.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m entitylinking.experiment \
        ${AIDA} \
        --entities=../data/embeddings/gigaword/entities.bin \
        --words=../data/embeddings/gigaword/words.bin \
        --ctxs=../data/embeddings/gigaword/words.bin \
        --minibatch-size=100 \
        --combo-method=sum \
        --predictions=${LOGF}.predictions \
        --use-mentions \
        -l ${LOGF} \
        1> /dev/null
fi
tail -n 1 ${LOGF}.predictions
