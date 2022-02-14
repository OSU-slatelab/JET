#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "vocab.h"
#include "io.h"
#include "logging.h"

void HeyISaw(struct vocabulary *vocab, char *word) {
    int ix = SearchVocab(vocab, word);
    if (ix == -1) ix = AddWordToVocab(vocab, word);
    IncrementVocabFreq(vocab, ix);
}

void ExtractWordVocabulary(char *f, struct vocabulary *vocab) {
    char *word = calloc(MAX_STRING, sizeof(char));
    FILE *fin = fopen(f, "rb");
    long long tokens_read = 0, last_printed_at = 0;

    // make EOS_TOKEN the first word in the vocab
    HeyISaw(vocab, EOSToken());

    ReadCorpusWord(word, fin, MAX_STRING);
    while (word[0] != -1) {
        HeyISaw(vocab, word);

        tokens_read++;
        if (tokens_read - last_printed_at > 1000000) {
            info("%c  >> Read %lld tokens  (%lld words in vocab)", 13,
                tokens_read, vocab->vocab_size);
            last_printed_at = tokens_read;
        }
        ReadCorpusWord(word, fin, MAX_STRING);
    }
    info("\n");
    fclose(fin);
}

void ExtractTermVocabulary(char *annotf, struct vocabulary *vocab) {
    struct term_annotation *annot = malloc(sizeof(struct term_annotation));
    annot->term = malloc(sizeof(struct indexed_string));
    annot->term->string = calloc(MAX_STRING, sizeof(char));

    FILE *fin = fopen(annotf, "rb");
    long long annotations_read = 0, last_printed_at = 0;

    while (!feof(fin)) {
        ReadAnnotation(annot, fin);

        HeyISaw(vocab, annot->term->string);

        annotations_read++;
        if (annotations_read - last_printed_at > 1000000) {
            info("%c  >> Read %lld annotations  (%lld terms in vocab)", 13,
                annotations_read, vocab->vocab_size);
            last_printed_at = annotations_read;
        }
    }
    info("\n");
    fclose(fin);

    if (annot != NULL && annot->term != NULL && annot->term->string != NULL)
        free(annot->term->string);
    if (annot != NULL && annot->term != NULL)
        free(annot->term);
    if (annot != NULL)
        free(annot);
}

struct vocabulary *MasterVocabHandler(char *corpusf, char *vocabf, char *type,
        bool read, bool overwrite, bool terms) {
    struct vocabulary *vocab = NULL;

    // get the previously-extracted vocabulary if it exists
    if (vocabf[0] != 0 && !overwrite && FileExists(vocabf)) {
        info("  Using %s vocab file %s...\n", type, vocabf);
        if (read) vocab = ReadVocab(vocabf);
    }
    // otherwise, extract it
    else {
        vocab = CreateVocabulary();
        info("  Learning %s vocabulary from %s...\n", type, corpusf);
        if (!terms)
            ExtractWordVocabulary(corpusf, vocab);
        else
            ExtractTermVocabulary(corpusf, vocab);
    }

    return vocab;
}

struct vocabulary *GetVocabulary(char *corpusf, char *vocabf, char *type, bool learning_tags) {
    return MasterVocabHandler(corpusf, vocabf, type, true, false, learning_tags);
}

struct vocabulary *LearnVocabulary(char *corpusf, char *vocabf, char *type, bool overwrite, bool learning_tags) {
    return MasterVocabHandler(corpusf, vocabf, type, false, overwrite, learning_tags);
}

struct vocabulary *GetWordVocabulary(char *plaintextf, char *vocabf) {
    return GetVocabulary(plaintextf, vocabf, "word", false);
}
struct vocabulary *GetTermVocabulary(char *annotationsf, char *vocabf) {
    return GetVocabulary(annotationsf, vocabf, "term", true);
}

struct vocabulary *LearnWordVocabulary(char *plaintextf, char *vocabf, bool overwrite) {
    return LearnVocabulary(plaintextf, vocabf, "word", overwrite, false);
}
struct vocabulary *LearnTermVocabulary(char *annotationsf, char *vocabf, bool overwrite) {
    return LearnVocabulary(annotationsf, vocabf, "term", overwrite, true);
}
