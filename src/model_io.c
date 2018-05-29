#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include "io.h"
#include "vocab.h"
#include "entities.h"
#include "logging.h"
#include "term_strings.h"
#include "monogamy.h"
#include "model.h"
#include "model_io.h"

/**
 * Write embeddings to a file, formatted as:
 *   First line: <Vocabulary size> <dimensionality>
 *   Remainder: <Embedded term> <feature 1> <feature 2> ...
 */
void WriteVectors(char *f, struct vocabulary *v, real *embeds, long long embed_size, int binary) {
    FILE *fo = fopen(f, "wb");
    // error check
    if (fo == NULL) {
        error("Unable to write to embedding file %s, check if the directory exists\n", f);
        exit(1);
    }
    // write header info (# words, # dimensions)
    fprintf(fo, "%ld %lld\n", v->vocab_size, embed_size);
    // write each embedding
    for (int a = 0; a < v->vocab_size; a++) {
        fprintf(fo, "%s ", v->vocab[a].word);
        if (binary) for (int b = 0; b < embed_size; b++) fwrite(&embeds[a * embed_size + b], sizeof(real), 1, fo);
        else for (int b = 0; b < embed_size; b++) fprintf(fo, "%lf ", embeds[a * embed_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

/**
 *
 */
void LoadVectors(char *fpath, real *embeddings, struct vocabulary *vocab, long long embed_size) {
    long read_vocab_size, embeds_read_sofar = 0;
    long long read_embed_size;
    char *embed_word = calloc(MAX_STRING, sizeof(char));
    bool reading_word = true, reading_embed = false;
    int current_word_ix = 0, word_vocab_ix;
    long long current_embed_ix = 0, embed_array_index;
    FILE *f = fopen(fpath, "rb");

    // error check
    if (f == NULL) {
        error("Unable to read embedding file %s\n", fpath);
        exit(1);
    }

    // confirm this is a valid embedding source
    fscanf(f, "%ld", &read_vocab_size);
    fscanf(f, "%lld", &read_embed_size);
    if (read_vocab_size != vocab->vocab_size) {
        error("Embedding file %s has vocab size %ld, expected vocab size %ld\n",
            fpath, read_vocab_size, vocab->vocab_size);
        exit(1);
    }
    if (read_embed_size != embed_size) {
        error("Embedding file %s has embedding size %lld, expected embedding size %lld\n",
            fpath, read_embed_size, embed_size);
        exit(1);
    }

    while (!feof(f)) {
        if (reading_word) {
            embed_word[current_word_ix] = fgetc(f);
            if (embed_word[current_word_ix] == ' ') {
                // switch to embed reading and reset for the next word
                reading_word = false;
                reading_embed = true;
                embed_word[current_word_ix] = 0;
                current_word_ix = 0;
                // get the index of this word in the vocabulary
                word_vocab_ix = SearchVocab(vocab, embed_word);
                if (word_vocab_ix == -1) {
                    error("Found embedding for unknown word: %s\n", embed_word);
                    exit(1);
                }
            } else { current_word_ix++; }
        }

        else if (reading_embed) {
            embed_array_index = (word_vocab_ix * embed_size) + current_embed_ix;
            fread(&embeddings[embed_array_index], sizeof(real), 1, f);
            current_embed_ix++;
            // if finished reading the embedding, switch back to word mode
            if (current_embed_ix == embed_size) {
                reading_embed = false;
                reading_word = true;
                current_embed_ix = 0;
                embeds_read_sofar++;
            }
        }
    }

    if (embeds_read_sofar != vocab->vocab_size) {
        error("Expected to read %ld embeddings, only read %ld\n",
            vocab->vocab_size, embeds_read_sofar);
        exit(1);
    }

    fclose(f);
}


/**
 * Write hyperparameters and all other training settings to file,
 * for replicability
 */
void WriteHyperparameters(char *f, struct hyperparameters params) {
    time_t now;
    struct tm ts;
    char time_buf[80];

    FILE *fo = fopen(f, "wb");
    if (fo == NULL) {
        error("Unable to write hyperparameters to file %s, check if the directory exists\n", f);
        exit(1);
    }

    time(&now);
    ts = *localtime(&now);
    strftime(time_buf, sizeof(time_buf), "%a %Y-%m-%d %H:%M:%S %Z", &ts);

    // model parameters
    fprintf(fo, "JET settings log\n");
    fprintf(fo, "Generated: %s\n", time_buf);
    fprintf(fo, "\n== Model parameters ==\n");
    fprintf(fo, "  # of iterations: %d\n", params.numiters);
    fprintf(fo, "  Window size: %d\n", params.window);
    fprintf(fo, "  Minimum frequency: %d\n", params.min_count);
    fprintf(fo, "  Embedding size: %lld\n", params.embedding_size);
    fprintf(fo, "  Alpha: %f\n", params.alpha);
    fprintf(fo, "  Alpha decay check interval: %lld\n", params.alpha_schedule_interval);
    fprintf(fo, "  Downsampling rate: %f\n", params.downsampling_rate);
    fprintf(fo, "  # of threads: %d\n", params.num_threads);
    fprintf(fo, "  Random seed: %ld\n", params.random_seed);

    // component flags
    fprintf(fo, "\n== Model components ==");
    fprintf(fo, "\n  Word learning: ");
    if (params.flags->disable_words) fprintf(fo, "DISABLED"); else fprintf(fo, "ENABLED");
    fprintf(fo, "\n  Term learning: ");
    if (params.flags->disable_terms) fprintf(fo, "DISABLED"); else fprintf(fo, "ENABLED");
    fprintf(fo, "\n  Entity learning: ");
    if (params.flags->disable_entities) fprintf(fo, "DISABLED"); else fprintf(fo, "ENABLED");
    fprintf(fo, "\n");

    // training files
    fprintf(fo, "\n== Training files ==\n");
    fprintf(fo, "  Plaintext corpus: %s\n", params.plaintext_corpus_file);
    fprintf(fo, "  Corpus annotations: %s\n", params.corpus_annotations_file);
    fprintf(fo, "  Term-entity map: %s\n", params.map_file);
    fprintf(fo, "  Term-entity map separator: %s\n", params.str_map_sep);
    fprintf(fo, "  Term-string map: %s\n", params.term_strmap_file);
    fprintf(fo, "  Thread configuration file: %s\n", params.thread_config_file);

    // vocabulary files
    fprintf(fo, "\n== Vocabulary files==\n");
    fprintf(fo, "  Word vocabulary: %s\n", params.wvocab_file);
    fprintf(fo, "  Term vocabulary: %s\n", params.tvocab_file);

    // model files
    fprintf(fo, "\n== Model files==\n");
    fprintf(fo, "  Word embeddings file: %s\n", params.word_vectors_file);
    fprintf(fo, "  Term embeddings file: %s\n", params.term_vectors_file);
    fprintf(fo, "  Entity embeddings file: %s\n", params.entity_vectors_file);
    fprintf(fo, "  Context embeddings file: %s\n", params.context_vectors_file);

    fclose(fo);
}
