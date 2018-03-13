#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "io.h"
#include "vocab.h"
#include "entities.h"
#include "term_strings.h"
#include "monogamy.h"
#include "model.h"
#include "logging.h"
#include "pre_init.h"

/**
 * Given a set of randomly-initialized term embeddings,
 * current word embeddings, and conditional term probabilities
 * calculated from the words, adjust each term to a weighted
 * combination of its words.
 *
 * Each term will retain some degree of its random initialization
 * (except for 1-word terms or otherwise completely predicted terms,
 * which will be copies of their words).
 */
void InitTermsFromWords(real *term_embeddings, struct vocabulary *term_vocab, 
        real *word_embeddings, struct vocabulary *word_vocab,
        struct term_string_map *strmap, struct term_monogamy_map *monomap,
        long long embedding_size) {
    int term_ix, word_ix, known_words;
    long long i, c, term_offset, word_offset;
    real combined_member_word_embedding[embedding_size];
    real word_monogamy_weight, preinit_weight;

    for (term_ix = 0; term_ix < term_vocab->vocab_size; term_ix++) {
        term_offset = term_ix * embedding_size;

        // initialize the member weighted sum to 0
        for (c = 0; c < embedding_size; c++)
            combined_member_word_embedding[c] = 0;

        // grab the weighted sum of the member words
        known_words = 0;
        preinit_weight = 0;
        for (i = 0; i < monomap->monogamies[term_ix].num_tokens; i++) {
            word_ix = SearchVocab(word_vocab, strmap->strings[term_ix].tokens[i]);
            if (word_ix >= 0) {
                word_offset = word_ix * embedding_size;
                word_monogamy_weight = monomap->monogamies[term_ix].by_word[i];
                preinit_weight += word_monogamy_weight;
                for (c = 0; c < embedding_size; c++) {
                    combined_member_word_embedding[c] +=
                        (word_embeddings[word_offset + c] * word_monogamy_weight);
                }
                known_words++;
            }
        }
        preinit_weight /= (real)known_words;

        for (c = 0; c < embedding_size; c++) {
            // keep (1-preinit_weight) of the random initialization
            term_embeddings[term_offset + c] *= (1 - preinit_weight);
            // and add in the rest from the member words (normalized by term length)
            term_embeddings[term_offset + c] += 
                (combined_member_word_embedding[c] / (real)known_words);
        }
    }
}
