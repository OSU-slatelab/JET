#!/bin/bash

PY=python3

source fetch_data.sh

DATA=../data/experiments/analogy

#############################################################
## BMASS - PubMed embeddings
#############################################################

if [ -e ${BMASS} ]; then
    echo
    echo "-- Results on UMNSRS with PubMed embeddings --"
    echo

    ENTLOGF=${DATA}/BMASS/results/pubmed_entities.log
    if [ ! -e ${ENTLOGF} ]; then
        echo "Running entity-based model..."
        ${PY} -m analogy.experiment \
            --dataset=BMASS \
            --setting=Multi-Answer \
            --type=concept \
            --predictions-file=${ENTLOGF}.predictions \
            -l ${ENTLOGF} \
            --entities=../data/embeddings/pubmed/entities.bin \
            --representation-method=ENTITY \
            ${BMASS}
    fi

    WRDLOGF=${DATA}/BMASS/results/pubmed_words.log
    if [ ! -e ${WRDLOGF} ]; then
        echo "Running word-based model..."
        ${PY} -m analogy.experiment \
            --dataset=BMASS \
            --setting=Multi-Answer \
            --type=string \
            --predictions-file=${WRDLOGF}.predictions \
            -l ${WRDLOGF} \
            --words=../data/embeddings/pubmed/words.bin \
            --representation-method=WORD \
            ${BMASS}
    fi

    # handle presence/absence of preferred strings file
    if [ -e ${UMLS_2017AB_PREFSTRS} ]; then
        PREFSTRLINE="--strings=${UMLS_2017AB_PREFSTRS}"
    else
        echo "[WARNING] Could not find ${UMLS_2017AB_PREFSTRS}; will use CUIs only in log output."
        echo
        PREFSTRLINE=
    fi
    # get performance/error analysis on paired results
    LOGF=${DATA}/BMASS/results/pubmed_comparison.log
    if [ ! -e ${LOGF} ]; then
        echo "Running error analysis..."
        ${PY} -m analogy.error_analysis \
            --output=${LOGF} \
            ${PREFSTRLINE} \
            --analogy-file=${BMASS} \
            ${ENTLOGF}.predictions CUI Entities \
            ${WRDLOGF}.predictions STR Words
    fi

    awk -F ',' '{
        if (NF == 4) {
            if (match($1, /B3:/) || match($1, /H1:/) || match($1, /C6:/) || match($1, /L1:/) || match($1, /L6:/)) {
                print $1 "\n   Entity Acc: " $2 "; Word acc: " $3 "; Oracle acc: " $4
            }
        }
    }' ${LOGF}

    echo
    echo "Full comparison of entity-based and word-based results can be found in"
    echo "   ${LOGF}"

else
    echo
    echo "Skipping BMASS analogies; need to download dataset"
    echo "File can be downloaded from this page:"
    echo "   http://slate.cse.ohio-state.edu/UTSAuthenticatedDownloader/index.html?dataset=BMASS"
fi
