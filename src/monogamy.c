#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include "vocab.h"
#include "io.h"
#include "logging.h"
#include "term_strings.h"
#include "monogamy.h"

struct term_monogamy_map *CreateMonogamyMap(long num_terms) {
    long i;

    struct term_monogamy_map *map = malloc(sizeof(struct term_monogamy_map));
    if (map == NULL) {
        error("   >>> Failed to allocate memory for term monogamy map; Aborting\n");
        exit(1);
    }

    map->monogamies = malloc(num_terms * sizeof(struct term_monogamy));
    if (map->monogamies == NULL) {
        error("   >>> Failed to allocate memory for monogamy scores in term monogamy map; Aborting\n");
        exit(1);
    }

    map->map_size = num_terms;

    for (i = 0; i < num_terms; i++) {
        map->monogamies[i].by_word = NULL;
        map->monogamies[i].monogamy_weight = 0;
    }

    return map;
}

void DestroyMonogamyMap(struct term_monogamy_map **map) {
    long i;
    if (*map != NULL) {
        if ((*map)->monogamies != NULL) {
            for (i = 0; i < (*map)->map_size; i++) {
                if ((*map)->monogamies[i].by_word != NULL) {
                    free((*map)->monogamies[i].by_word);
                    (*map)->monogamies[i].by_word = NULL;
                }
            }
            free((*map)->monogamies);
            (*map)->monogamies = NULL;
        }
        free(*map);
        *map = NULL;
    }
}

void CalculateMonogamy(struct vocabulary *term_vocab, struct vocabulary *word_vocab,
        struct term_string_map *strmap, long long unigram_corpus_size,
        struct term_monogamy_map *map) {
    long term_ix, word_ix;
    long long ngram_corpus_size;
    float word_probability, term_probability, sum_monogamy;
    int num_tokens, valid_tokens, i;

    for (term_ix = 0; term_ix < term_vocab->vocab_size; term_ix++) {
        num_tokens = strmap->strings[term_ix].num_tokens;

        // adjust corpus token count for ngram size and calculate term prob
        ngram_corpus_size = unigram_corpus_size - num_tokens + 1;
        term_probability = term_vocab->vocab[term_ix].cn / (float)ngram_corpus_size;

        // grab memory for the scores
        map->monogamies[term_ix].by_word = malloc(num_tokens * sizeof(float));
        if (map->monogamies[term_ix].by_word == NULL) {
            error("   >>> Failed to allocate memory for term-word scores; Aborting\n");
            exit(1);
        }
        // and calculate scores
        valid_tokens = 0;
        sum_monogamy = 0;
        for (i = 0; i < num_tokens; i++) {
            word_ix = SearchVocab(word_vocab, strmap->strings[term_ix].tokens[i]);
            if (word_ix >= 0) {
                word_probability = word_vocab->vocab[word_ix].cn / (float)unigram_corpus_size;
                map->monogamies[term_ix].by_word[i] = (term_probability / word_probability);
                sum_monogamy += map->monogamies[term_ix].by_word[i];
                valid_tokens++;
            } else {
                map->monogamies[term_ix].by_word[i] = 0;
            }
        }
        map->monogamies[term_ix].num_tokens = num_tokens;

        // mean the observed word scores
        if (valid_tokens > 0) {
            map->monogamies[term_ix].monogamy_weight = sum_monogamy / (float)valid_tokens;
        } else {
            map->monogamies[term_ix].monogamy_weight = 0;
        }
    }
}
