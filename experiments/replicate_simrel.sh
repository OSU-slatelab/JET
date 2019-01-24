#!/bin/bash

PY=python3
PYTHONPATH=$(pwd)/../dependencies

source fetch_data.sh

DATA=../data/experiments/simrel

#############################################################
## UMNSRS with PubMed embeddings
#############################################################

echo
echo "-- Results on UMNSRS with PubMed embeddings --"

if [ -e "${BL_CHIU_2016}" ]; then
    echo "Baseline: Chiu et al (2016)..."
    LOGF=${DATA}/UMNSRS/results/pubmed_baseline_chiu-et-al-2016.log
    if [ ! -e ${LOGF} ]; then
        ${PY} -m simrel.experiment \
            --mode=UMNSRS \
            --representation-method=WORD \
            --words=${BL_CHIU_2016} \
            --keep-word-case \
            -l ${LOGF} \
            1>/dev/null
    fi
    tail -n 2 ${LOGF}
else
    echo "Chiu et al (2016) baseline not found"
    echo "  To download, please visit https://github.com/cambridgeltl/BioNLP-2016"
fi

echo "Baseline: DeVine et al (2014)..."
LOGF=${DATA}/UMNSRS/results/pubmed_baseline_devine-et-al-2014.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=UMNSRS \
        --representation-method=ENTITY \
        --entities=${BL_DEVINE_2014} \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "Baseline: Mencia et al (2016)..."
LOGF=${DATA}/UMNSRS/results/pubmed_baseline_mencia-et-al-2016.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=UMNSRS \
        --representation-method=ENTITY \
        --entities=${BL_MENCIA_2016} \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Words..."
LOGF=${DATA}/UMNSRS/results/pubmed_words.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=UMNSRS \
        --representation-method=WORD \
        --words=../data/embeddings/pubmed/words.bin \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Terms..."
LOGF=${DATA}/UMNSRS/results/pubmed_terms.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=UMNSRS \
        --representation-method=TERM \
        --terms=../data/embeddings/pubmed/terms.bin \
        --words=../data/embeddings/pubmed/words.bin \
        --string-map=../data/terminologies/UMLS_2017AB/term-string-map.txt \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Entity..."
LOGF=${DATA}/UMNSRS/results/pubmed_entities.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=UMNSRS \
        --representation-method=ENTITY \
        --entities=../data/embeddings/pubmed/entities.bin \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Entities+Words..."
LOGF=${DATA}/UMNSRS/results/pubmed_entity-word-combo_no-cross.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=UMNSRS \
        --entities=../data/embeddings/pubmed/entities.bin \
        --words=../data/embeddings/pubmed/words.bin \
        --combo \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Entities+Words+Cross..."
LOGF=${DATA}/UMNSRS/results/pubmed_entity-word-combo_cross.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=UMNSRS \
        --entities=../data/embeddings/wikipedia/entities.bin \
        --words=../data/embeddings/wikipedia/words.bin \
        --combo \
        --cross \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}


#############################################################
## UMNSRS filtered subset with PubMed embeddings
#############################################################

if [ -e ${BL_DEVINE_2014} ] && [ -e ${BL_MENCIA_2016} ]; then

    echo
    echo "-- Results on UMNSRS (filtered subset) with PubMed embeddings --"

    echo "Baseline (filtered): DeVine et al (2014)..."
    LOGF=${DATA}/UMNSRS/results/pubmed_filtered_baseline_devine-et-al-2014.log
    if [ ! -e ${LOGF} ]; then
        ${PY} -m simrel.experiment \
            --mode=UMNSRS \
            --representation-method=ENTITY \
            --entities=${BL_DEVINE_2014} \
            --skip-indices=${UMNSRS_FILTER} \
            -l ${LOGF} \
            1>/dev/null
    fi
    tail -n 2 ${LOGF}

    echo "Baseline (filtered): Mencia et al (2016)..."
    LOGF=${DATA}/UMNSRS/results/pubmed_filtered_baseline_mencia-et-al-2016.log
    if [ ! -e ${LOGF} ]; then
        ${PY} -m simrel.experiment \
            --mode=UMNSRS \
            --representation-method=ENTITY \
            --entities=${BL_MENCIA_2016} \
            --skip-indices=${UMNSRS_FILTER} \
            -l ${LOGF} \
            1>/dev/null
    fi
    tail -n 2 ${LOGF}

    echo "JET (filtered): Entity..."
    LOGF=${DATA}/UMNSRS/results/pubmed_filtered_entities.log
    if [ ! -e ${LOGF} ]; then
        ${PY} -m simrel.experiment \
            --mode=UMNSRS \
            --representation-method=ENTITY \
            --entities=../data/embeddings/pubmed/entities.bin \
            --skip-indices=${UMNSRS_FILTER} \
            -l ${LOGF} \
            1>/dev/null
    fi
    tail -n 2 ${LOGF}

    echo "JET (filtered): Entities+Words..."
    LOGF=${DATA}/UMNSRS/results/pubmed_filtered_entity-word-combo_no-cross.log
    if [ ! -e ${LOGF} ]; then
        ${PY} -m simrel.experiment \
            --mode=UMNSRS \
            --entities=../data/embeddings/pubmed/entities.bin \
            --words=../data/embeddings/pubmed/words.bin \
            --skip-indices=${UMNSRS_FILTER} \
            --combo \
            -l ${LOGF} \
            1>/dev/null
    fi
    tail -n 2 ${LOGF}

    echo "JET (filtered): Entities+Words+Cross..."
    LOGF=${DATA}/UMNSRS/results/pubmed_filtered_entity-word-combo_cross.log
    if [ ! -e ${LOGF} ]; then
        ${PY} -m simrel.experiment \
            --mode=UMNSRS \
            --entities=../data/embeddings/pubmed/entities.bin \
            --words=../data/embeddings/pubmed/words.bin \
            --skip-indices=${UMNSRS_FILTER} \
            --combo \
            --cross \
            -l ${LOGF} \
            1>/dev/null
    fi
    tail -n 2 ${LOGF}

else
    echo
    echo "Skipping UMNSRS filtered subset"
    echo "  Need both DeVine et al (2014) and Mencia et al (2016) baselines"
fi


#############################################################
## WikiSRS with Wikipedia embeddings
#############################################################

echo
echo "-- Results on WikiSRS with Wikipedia embeddings --"

echo "Baseline: word2vec..."
LOGF=${DATA}/WikiSRS/results/wikipedia_baseline_word2vec.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --representation-method=WORD \
        --words=../data/embeddings/wikipedia/word2vec_baseline.bin \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

if [ -e "${BL_WIKI_MPME}" ]; then
    echo "Baseline: Cao et al (2017)..."
    LOGF=${DATA}/WikiSRS/results/wikipedia_baseline_cao-et-al-2017.log
    if [ ! -e ${LOGF} ]; then
        ${PY} -m simrel.experiment \
            --mode=WikiSRS \
            --representation-method=ENTITY \
            --entities=${BL_WIKI_MPME} \
            --tab-separated \
            -l ${LOGF} \
            1>/dev/null
    fi
    tail -n 2 ${LOGF}
else
    echo "Cao et al (2017) relearned Wikipedia baseline not found"
    echo "  To download, please ... [do something TBD]"
fi

echo "JET: Words..."
LOGF=${DATA}/WikiSRS/results/wikipedia_words.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --representation-method=WORD \
        --words=../data/embeddings/wikipedia/words.bin \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Terms..."
LOGF=${DATA}/WikiSRS/results/wikipedia_terms.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --representation-method=TERM \
        --terms=../data/embeddings/wikipedia/terms.bin \
        --words=../data/embeddings/wikipedia/words.bin \
        --string-map=../data/terminologies/Wikipedia_20180120/term-string-map.txt \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Entities..."
LOGF=${DATA}/WikiSRS/results/wikipedia_entities.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --representation-method=ENTITY \
        --entities=../data/embeddings/wikipedia/entities.bin \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Entities+Words..."
LOGF=${DATA}/WikiSRS/results/wikipedia_entity-word-combo_no-cross.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --entities=../data/embeddings/wikipedia/entities.bin \
        --words=../data/embeddings/wikipedia/words.bin \
        --combo \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Entities+Words+Cross..."
LOGF=${DATA}/WikiSRS/results/wikipedia_entity-word-combo_cross.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --entities=../data/embeddings/wikipedia/entities.bin \
        --words=../data/embeddings/wikipedia/words.bin \
        --combo \
        --cross \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}


#############################################################
## WikiSRS with Gigaword embeddings
#############################################################

echo
echo "-- Results on WikiSRS with Gigaword embeddings --"

echo "Baseline: word2vec..."
LOGF=${DATA}/WikiSRS/results/gigaword_baseline_word2vec.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --representation-method=WORD \
        --words=../data/embeddings/gigaword/word2vec_baseline.bin \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Words..."
LOGF=${DATA}/WikiSRS/results/gigaword_words.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --representation-method=WORD \
        --words=../data/embeddings/gigaword/words.bin \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Terms..."
LOGF=${DATA}/WikiSRS/results/gigaword_terms.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --representation-method=TERM \
        --terms=../data/embeddings/gigaword/terms.bin \
        --words=../data/embeddings/gigaword/words.bin \
        --string-map=../data/terminologies/Wikipedia_20180120/term-string-map.txt \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Entities..."
LOGF=${DATA}/WikiSRS/results/gigaword_entities.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --representation-method=ENTITY \
        --entities=../data/embeddings/gigaword/entities.bin \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Entities+Words..."
LOGF=${DATA}/WikiSRS/results/gigaword_entity-word-combo_no-cross.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --entities=../data/embeddings/gigaword/entities.bin \
        --words=../data/embeddings/gigaword/words.bin \
        --combo \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}

echo "JET: Entities+Words+Cross..."
LOGF=${DATA}/WikiSRS/results/gigaword_entity-word-combo_cross.log
if [ ! -e ${LOGF} ]; then
    ${PY} -m simrel.experiment \
        --mode=WikiSRS \
        --entities=../data/embeddings/gigaword/entities.bin \
        --words=../data/embeddings/gigaword/words.bin \
        --combo \
        --cross \
        -l ${LOGF} \
        1>/dev/null
fi
tail -n 2 ${LOGF}
