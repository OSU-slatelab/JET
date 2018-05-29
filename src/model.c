#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include "logging.h"
#include "mem.h"
#include "cli.h"
#include "io.h"
#include "vocab.h"
#include "entities.h"
#include "parallel_reader.h"
#include "thread_config.h"
#include "vocab_learner.h"
#include "term_strings.h"
#include "monogamy.h"
#include "model.h"
#include "mt19937ar.h"

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define TERM_SAMPLE_FACTOR 3
#define NORM_LIMIT 500

real *exp_table;
const int unigram_table_size = 1e8;

/**
 * Calculate the dot product of two embeddings
 */
real DotProduct(real *embeds_a, long long offset_a, real *embeds_b, long long offset_b,
        long long embedding_size) {
    long long c;
    real dot = 0;
    for (c = 0; c < embedding_size; c++)
        dot += embeds_a[offset_a + c] * embeds_b[offset_b + c];
    return dot;
}

/**
 * Calculate the L2-norm of an embeddings
 */
real Norm(real *embeds_a, long long offset_a, long long embedding_size) {
    real norm = 0;
    for (long long c = 0; c < embedding_size; c++) norm += (real)pow(embeds_a[c + offset_a], 2);
    if (norm > 0)
        return (real)sqrt(norm);
    else
        return 0;
}

/**
 * Calculate cosine from pre-dotted vectors
 */
real CosineSimilarityFromDot(real dot_product, real norm_a, real norm_b) {
    return (
        dot_product /
        (norm_a * norm_b)
    );
}

/**
 * Calculate the cosine similarity between two vectors
 */
real CosineSimilarity(real *embeds_a, long long offset_a, real *embeds_b,
        long long offset_b, real norm_a, real norm_b, long long embedding_size) {

    if (norm_a == 0 || norm_b == 0)
        return 0;

    real dot_product = DotProduct(embeds_a, offset_a, embeds_b, offset_b, embedding_size);
    return CosineSimilarityFromDot(dot_product, norm_a, norm_b);
}

/**
 * Allocate and initialize values for model flags
 */
void InitModelFlags(struct model_flags **flags) {
    *flags = malloc(sizeof(struct model_flags));
    if (*flags == NULL) {
        error("   >>> Failed to initialize model flags; Aborting\n");
        exit(1);
    }
    (*flags)->disable_words = false;
    (*flags)->disable_terms = false;
    (*flags)->disable_entities = false;
}
/**
 * Destroy model flags
 */
void DestroyModelFlags(struct model_flags **flags) {
    if (*flags != NULL) {
        free(*flags);
        *flags = NULL;
    }
}

/**
 * Pre-calculate the exp() table (for gradient/output computation)
 *
 * @see ComputeGradient for how this is actually used
 */
void InitExpTable() {
    exp_table = malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (long i = 0; i < EXP_TABLE_SIZE; i++) {
        exp_table[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);   // Precompute the exp() table
        exp_table[i] = exp_table[i] / (exp_table[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
}
/**
 * Destroy the exp() table
 */
void DestroyExpTable() {
    if (exp_table != NULL) {
        free(exp_table);
        exp_table = NULL;
    }
}

/**
 * Initialize all embedding matrices: word, tag, concept, and negative sampling
 * TODO: rename to InitEmbeddings
 */
void InitNet(real **word_embeddings, real **term_embeddings, real **entity_embeddings, real **ctx_embeddings,
        real **word_norms, real **term_norms, real **entity_norms, real **ctx_norms,
        struct vocabulary *wv, struct vocabulary *tv, struct vocabulary *ev, long long embedding_size) {
    long long a, b;

    // initialize word embeddings
    *word_embeddings = malloc(wv->vocab_size * embedding_size * sizeof(real));
    if (*word_embeddings == NULL) {
        error("Memory allocation failed for word embeddings\n");
        exit(1);
    }
    for (b = 0; b < embedding_size; b++) {
        for (a = 0; a < wv->vocab_size; a++) {
            (*word_embeddings)[a * embedding_size + b] = (genrand_real3() - 0.5) / (real)embedding_size;
        }
    }

    // initialize term embeddings
    *term_embeddings = malloc(tv->vocab_size * embedding_size * sizeof(real));
    if (*term_embeddings == NULL) {
        error("Memory allocation failed for tag embeddings\n");
        exit(1);
    }
    for (b = 0; b < embedding_size; b++) {
        for (a = 0; a < tv->vocab_size; a++) {
            (*term_embeddings)[a * embedding_size + b] = (genrand_real3() - 0.5) / (real)embedding_size;
        }
    }

    // initialize entity embeddings
    *entity_embeddings = malloc(ev->vocab_size * embedding_size * sizeof(real));
    if (*entity_embeddings == NULL) {
        error("Memory allocation failed for concept embeddings\n");  
        exit(1);
    }
    for (b = 0; b < embedding_size; b++) {
        for (a = 0; a < ev->vocab_size; a++) {
            (*entity_embeddings)[a * embedding_size + b] = (genrand_real3() - 0.5) / (real)embedding_size;
        }
    }

    // initialize negative sample embeddings from the word vocab
    *ctx_embeddings = malloc(wv->vocab_size * embedding_size * sizeof(real));
    if (*ctx_embeddings == NULL) {
        error("Memory allocation failed for negative sample embeddings\n");
        exit(1);
    }
    for (b = 0; b < embedding_size; b++) {
        for (a = 0; a < wv->vocab_size; a++) {
            (*ctx_embeddings)[a * embedding_size + b] = (genrand_real3() - 0.5) / (real)embedding_size;
        }
    }
    //*ctx_embeddings = *word_embeddings;

    // initialize norms
    *word_norms = malloc(wv->vocab_size * sizeof(real));
    if (*word_norms == NULL) {
        error("Memory allocation failed for word norms\n");
        exit(1);
    }
    for (a = 0; a < wv->vocab_size; a++) {
        (*word_norms)[a] = Norm(*word_embeddings, a * embedding_size, embedding_size);
    }

    *term_norms = malloc(tv->vocab_size * sizeof(real));
    if (*term_norms == NULL) {
        error("Memory allocation failed for term norms\n");
        exit(1);
    }
    for (a = 0; a < tv->vocab_size; a++) {
        (*term_norms)[a] = Norm(*term_embeddings, a * embedding_size, embedding_size);
    }

    *entity_norms = malloc(ev->vocab_size * sizeof(real));
    if (*entity_norms == NULL) {
        error("Memory allocation failed for entity norms\n");
        exit(1);
    }
    for (a = 0; a < ev->vocab_size; a++) {
        (*entity_norms)[a] = Norm(*entity_embeddings, a * embedding_size, embedding_size);
    }

    *ctx_norms = malloc(wv->vocab_size * sizeof(real));
    if (ctx_norms == NULL) {
        error("Memory allocation failed for context norms\n");
        exit(1);
    }
    for (a = 0; a < wv->vocab_size; a++) {
        (*ctx_norms)[a] = Norm(*ctx_embeddings, a * embedding_size, embedding_size);
    }
    //*ctx_norms = *word_norms;
}
/**
 * Destroy the embedding and norm arrays
 */
void DestroyNet(real **word_embeddings, real **term_embeddings, real **entity_embeddings, real **ctx_embeddings,
        real **word_norms, real **term_norms, real **entity_norms, real **ctx_norms) {
    if (*word_embeddings != NULL) {
        free(*word_embeddings);
        *word_embeddings = NULL;
    }
    if (*term_embeddings != NULL) {
        free(*term_embeddings);
        *term_embeddings = NULL;
    }
    if (*entity_embeddings != NULL) {
        free(*entity_embeddings);
        *entity_embeddings = NULL;
    }
    if (*ctx_embeddings != NULL) {
        free(*ctx_embeddings);
        *ctx_embeddings = NULL;
    }
    if (*word_norms != NULL) {
        free(*word_norms);
        *word_norms = NULL;
    }
    if (*term_norms != NULL) {
        free(*term_norms);
        *term_norms = NULL;
    }
    if (*entity_norms != NULL) {
        free(*entity_norms);
        *entity_norms = NULL;
    }
    if (*ctx_norms != NULL) {
        free(*ctx_norms);
        *ctx_norms = NULL;
    }
}

/**
 * Set up for sampling of negative examples.
 * wc[i] == the count of context number i
 * wclen is the number of entries in wc (context vocab size)
 */
void InitUnigramTable(int **unitable, struct vocabulary *v) {
    int a, i;
    long long normalizer = 0;
    real d1, power = 0.75;
    *unitable = malloc(unigram_table_size * sizeof(int));
    if (*unitable == NULL) {
        error("   >>> Failed to allocate memory for downsampling table; Aborting\n");
        exit(1);
    }
    for (a = 0; a < v->vocab_size; a++) normalizer += pow(v->vocab[a].cn, power);
    i = 0;
    d1 = pow(v->vocab[i].cn, power) / (real)normalizer;
    for (a = 0; a < unigram_table_size; a++) {
        (*unitable)[a] = i;
        if (a / (real)unigram_table_size > d1) {
            i++;
            d1 += pow(v->vocab[i].cn, power) / (real)normalizer;
        }
        if (i >= v->vocab_size) i = v->vocab_size - 1;
    }
}
/**
 * Destroy the unigram sampling table
 */
void DestroyUnigramTable(int **unitable) {
    if (*unitable != NULL) {
        free(*unitable);
        *unitable = NULL;
    }
}

/**
 * Set up for downsampling frequent words (includes memory
 * allocation for table).
 * Let 
 *  - f(w) be the frequency of word w
 *  - t be the downsampling rate
 *  - Z be the # of tokens in the corpus
 * Then, the likelihood of downsampling word w is:
 *   P(w) = max(0, 1 - sqrt((t*Z)/f(w)) + ((t*Z)/f(w)))
 */
void InitDownsamplingTable(real **downsampling_table, struct vocabulary *v, real downsampling_rate) {
    int i;
    long long corpus_size = v->word_count;
    real downsample_score;

    *downsampling_table = malloc(v->vocab_size * sizeof(real));
    if (*downsampling_table == NULL) {
        error("   >>> Failed to allocate memory for downsampling table; Aborting\n");
        exit(1);
    }
    for (i = 0; i < v->vocab_size; i++) {
        // calculate the score
        downsample_score = 1 - (
            sqrt(
                (downsampling_rate * corpus_size) /
                v->vocab[i].cn
            ) + (
                (downsampling_rate * corpus_size) /
                v->vocab[i].cn
            )
        );
        // clip at 0 so we can call it a probability
        if (downsample_score < 0) downsample_score = 0;
        // and save it
        (*downsampling_table)[i] = downsample_score;
    }
}
/**
 * Destroy the downsampling rate table
 */
void DestroyDownsamplingTable(real **downsampling_table) {
    if (*downsampling_table != NULL) {
        free(*downsampling_table);
        *downsampling_table = NULL;
    }
}

/**
 * Wrapper for all model initialization steps
 */
void InitializeModel(real **word_embeddings, real **term_embeddings, real **entity_embeddings,
        real **ctx_embeddings, real **word_norms, real **term_norms, real **entity_norms,
        real **ctx_norms,
        struct vocabulary *wv, struct vocabulary *tv, struct vocabulary *ev, struct entity_map *em,
        long long embedding_size, int **unitable, real **word_downsampling_table,
        real **term_downsampling_table, real downsampling_rate) {
    InitExpTable();
    InitNet(word_embeddings, term_embeddings, entity_embeddings, ctx_embeddings,
        word_norms, term_norms, entity_norms, ctx_norms, wv, tv, ev, embedding_size);
    InitUnigramTable(unitable, wv);
    InitDownsamplingTable(word_downsampling_table, wv, downsampling_rate);
    InitDownsamplingTable(term_downsampling_table, tv, downsampling_rate * TERM_SAMPLE_FACTOR);
}
/**
 * Wrapper for all model destruction steps
 */
void DestroyModel(real **word_embeddings, real **term_embeddings, real **entity_embeddings,
        real **ctx_embeddings, real **word_norms, real **term_norms, real **entity_norms,
        real **ctx_norms,
        int **unitable, real **word_downsampling_table, real **term_downsampling_table) {
    DestroyExpTable();
    DestroyNet(word_embeddings, term_embeddings, entity_embeddings, ctx_embeddings,
               word_norms, term_norms, entity_norms, ctx_norms);
    DestroyUnigramTable(unitable);
    DestroyDownsamplingTable(word_downsampling_table);
    DestroyDownsamplingTable(term_downsampling_table);
}

/**
 * For a given word or term index, roll a random number and
 * check against the pre-calculated downsampling threshold to
 * see if this instance should be downsampled (ignored).
 */
bool RollToDownsample(real *downsampling_table, int ix) {
    if (downsampling_table[ix] > genrand_real3())
        return true;
    else
        return false;
}

/**
 * Calculate a random start index in the left context of the window;
 * Idea is to observe closer words more often (decrease impact of distant
 * words in the training)
 */
int RandomSubwindowSkip(int window_size) {
    return genrand_int32() % window_size;
}

/**
 * Calculate the outer gradient the sigmoid function, using only the
 * pre-calculated dot product.  (Inner gradient of the dot product
 * itself is handled in calling functions.)
 *
 * Let x be the dot product of vectors v_a and v_b.  Then the
 * gradient d/dx of the sigmoid is
 * 
 * if label == 1:
 *    g = d/dx(log(1/(1 + e^-x)))
 *      = 1/(e^x + 1)
 *
 * if label == 0:
 *    g = d/dx(log(1/1 + e^-(-x))))
 *      = -e^x/(e^x + 1)
 */
real CalculateLogSigmoidOuterGradient(real dot_product, int label) {
    real gradient;
    // calculate (bounded) penalty multiplier
    if (dot_product > MAX_EXP) gradient = (label - 1);
    else if (dot_product < -MAX_EXP) gradient = (label - 0);
    else gradient = (label - exp_table[(int)((dot_product + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]);
    return gradient;
}


/**
 * Given a set of completed terms, calculate the weighted sum
 * of the embeddings of each of their sets of member words
 * [weighted by P(t|w)].
 */
void CombineWeightedMemberWordEmbeddings(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, real *word_embeddings,
        struct term_monogamy_map *monomap, long long embedding_size,
        real *combined_word_embeddings) {
    long long c, word_offset, combined_offset;
    int i, j, word_ix, term_ix, known_words;
    float mono_weight;

    for (i = 0; i < num_completed_terms; i++) {
        combined_offset = i * embedding_size;
        known_words = 0;

        // initialize to 0
        for (c = 0; c < embedding_size; c++)
            combined_word_embeddings[combined_offset + c] = 0;

        // if we're using the term, then average its member words
        term_ix = sampled_completed_term_ixes[i];
        if (term_ix >= 0) {
            // sum embeddings of all known words
            for (j = 0; j < completed_term_buffer[i]->num_tokens; j++) {
                word_ix = completed_term_buffer[i]->member_words[j];
                if (word_ix > 0) {
                    word_offset = word_ix * embedding_size;
                    mono_weight = monomap->monogamies[term_ix].by_word[j];
                    for (c = 0; c < embedding_size; c++) {
                        combined_word_embeddings[combined_offset + c]
                            += (mono_weight * word_embeddings[word_offset + c]);
                    }
                    known_words++;
                }
            }
            // normalize
            if (known_words > 0) {
                for (c = 0; c < embedding_size; c++)
                    combined_word_embeddings[combined_offset + c] /= known_words;
            }
        }
    }
}


/**
 * Given a set of completed terms, calculate the current average
 * embeddings of their context words.
 */
void CalculateAverageContextEmbeddings(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, real *ctx_embeddings,
        long long embedding_size, int window_start, int window_end, int target,
        real *averaged_ctx_embeddings) {
    long long c, ctx_offset, average_offset;
    int i, j, ctx_ix, known_words;

    for (i = 0; i < num_completed_terms; i++) {
        average_offset = i * embedding_size;
        known_words = 0;

        // initialize to 0
        for (c = 0; c < embedding_size; c++)
            averaged_ctx_embeddings[average_offset + c] = 0;

        // if we're using the term, then average its contexts
        if (sampled_completed_term_ixes[i] >= 0) {
            // sum embeddings of all known words
            for (j = window_start; j < window_end; j++) {
                if (j == target) continue;
                ctx_ix = completed_term_buffer[i]->contexts[j];
                if (ctx_ix >= 0) {
                    ctx_offset = ctx_ix * embedding_size;
                    for (c = 0; c < embedding_size; c++) {
                        averaged_ctx_embeddings[average_offset + c]
                            += ctx_embeddings[ctx_offset + c];
                    }
                    known_words++;
                }
            }
            // normalize
            if (known_words > 0) {
                for (c = 0; c < embedding_size; c++)
                    averaged_ctx_embeddings[average_offset + c] /= known_words;
            }
        }
    }
}


/**
 * Track the set of context words used as both positive and
 * negative samples for this training batch.
 */
void TrackContextIndices(int *masked_word_context_window, int word_ix,
        int *word_negative_samples, struct term_annotation **completed_term_buffer,
        int *term_negative_samples, int num_completed_terms,
        int *sampled_completed_term_ixes, int window_start, int window_end,
        int full_window_size, int target, int negative, int *all_ctx_ixes,
        int num_ctx, struct model_flags *flags) {
    int a, i, d, ctx_ix, word_ctx_block_start, term_ctx_block_start;
    int all_ctx_ixes_ix = 0;
    
    // first, set all ixes to -1 (unused)
    for (i = 0; i < num_ctx; i++) all_ctx_ixes[i] = -1;

    for (a = window_start; a < window_end; a++) {
        if (a == target) continue;

        // grab contexts used by the current word
        if (!flags->disable_words) {
            if (word_ix >= 0) {
                ctx_ix = masked_word_context_window[a];
                if (ctx_ix >= 0)
                    all_ctx_ixes[all_ctx_ixes_ix++] = ctx_ix;

                for (d = 0; d < negative; d++) {
                    word_ctx_block_start = a * negative;
                    ctx_ix = word_negative_samples[word_ctx_block_start + d];
                    if (ctx_ix >= 0)
                        all_ctx_ixes[all_ctx_ixes_ix++] = ctx_ix;
                }
            }
        }

        // grab contexts used by terms/entities
        if (!flags->disable_terms || !flags->disable_entities) {
            for (i = 0; i < num_completed_terms; i++) {
                if (sampled_completed_term_ixes[i] >= 0) {
                    // positive sample
                    ctx_ix = completed_term_buffer[i]->contexts[a];
                    if (ctx_ix >= 0)
                        all_ctx_ixes[all_ctx_ixes_ix++] = ctx_ix;

                    // negative samples
                    for (d = 0; d < negative; d++) {
                        term_ctx_block_start = ((i * full_window_size) + a) * negative;
                        ctx_ix = term_negative_samples[term_ctx_block_start + d];
                        if (ctx_ix >= 0)
                            all_ctx_ixes[all_ctx_ixes_ix++] = ctx_ix;
                    }
                }
            }
        }
    }
}



/**
 * Randomly selects a single word different from pos_ctx_ix, sampling
 * based on observed frequencies.
 */
int GetNegativeSample(int pos_ctx_ix, int *unitable) {
    unsigned long random;
    int neg_ctx_ix = pos_ctx_ix;
    if (pos_ctx_ix < 0) {
        neg_ctx_ix = -1;
    } else {
        while (neg_ctx_ix == 0 || neg_ctx_ix == pos_ctx_ix) {
            random = genrand_int32();
            neg_ctx_ix = unitable[random % unigram_table_size];
        }
    }
    return neg_ctx_ix;
}
/**
 * Fill a window of negative samples, given an observed window of positive
 * samples.
 */
void GetNegativeSamples(int negative, int *negative_samples, int ns_start_ix, int *pos_ctx_ixes,
        int window_start, int window_end, int target, int *unitable) {
    int i, j, window_block_start;
    for (i = window_start; i < window_end; i++) {
        window_block_start = i * negative;
        if (i == target) {
            for (j = 0; j < negative; j++)
                negative_samples[ns_start_ix + window_block_start + j] = -1;
        }
        else {
            for (j = 0; j < negative; j++)
                negative_samples[ns_start_ix + window_block_start + j] =
                    GetNegativeSample(pos_ctx_ixes[i], unitable);
        }
    }
}


/**
 * Given a specific target embedding, calculates its dot product with
 * each positive (observed) context word and each negative (non-observed)
 * sample, and stores them in pos_ctx_dots and neg_ctx_dots.
 */
void CalculateContextDotProducts(real *embeddings, long long trg_offset,
        real *ctx_embeddings, int *context_window, int *negative_samples,
        int negative, long ns_start_ix, int full_window_size, int target,
        int window_start, int window_end, long long embedding_size,
        real *pos_ctx_dots, long long pos_ctx_start_ix,
        real *neg_ctx_dots, long long neg_ctx_start_ix) {

    int a, d, ctx_ix;

    // zero out everything first
    for (a = 0; a < full_window_size; a++) {
        pos_ctx_dots[pos_ctx_start_ix + a] = 0;
        for (d = 0; d < negative; d++)
            neg_ctx_dots[neg_ctx_start_ix + (a*negative) + d] = 0;
    }

    for (a = window_start; a < window_end; a++) {
        // skip the target word
        if (a == target) continue;
        // if this context word is unknown, ignore it and move on
        if (context_window[a] < 0) continue;

        // calculate dot product with this positive context
        pos_ctx_dots[pos_ctx_start_ix + a] = DotProduct(embeddings, trg_offset,
            ctx_embeddings, context_window[a] * embedding_size, embedding_size);
        // and with each of the negative samples
        for (d = 0; d < negative; d++) {
            ctx_ix = negative_samples[ns_start_ix + (a * negative) + d];
            neg_ctx_dots[neg_ctx_start_ix + (a * negative) + d] = DotProduct(
                embeddings, trg_offset, ctx_embeddings,
                ctx_ix * embedding_size, embedding_size
            );
        }
    }
}

/**
 * Calculates word/term/entity gradients based on observed (positive) and
 * random (negative) context words, and adds to the current gradient vector.
 *
 * Let W_P be the observed contexts, and W_N be negative samples.  Then,
 * the gradients of target embedding e and context embeddings e_ctx are
 * calculated as:
 *
 *   up_e <- \sum_{w \in W_P} grad(e . ctx_w, 1) * ctx_w
 *           + \sum_{w \in W_N} grad(e . ctx_w, 0) * ctx_w
 *
 *   \forall w \in W_P : up_w <- grad(e . ctx_w, 1) * e
 *   \forall w \in W_N : up_w <- grad(e . ctx_w, 0) * e
 *
 * The constant_weight input parameter allows for arbitrary weighting of
 * these gradients (gradients are multiplied by constant_weight).
 */
void AddContextBasedGradients(real *embeddings, long long trg_offset,
        real *ctx_embeddings, int *context_window,
        int *negative_samples, int negative, long ns_start_ix,
        int full_window_size, int target, int window_start, int window_end,
        real *pos_ctx_dots, long long pos_ctx_start_ix,
        real *neg_ctx_dots, long long neg_ctx_start_ix,
        real *gradients, long gradient_start_ix, real *pos_ctx_gradients,
        long pos_ctx_gradient_start_ix, real *neg_ctx_gradients,
        long neg_ctx_gradient_start_ix, long long embedding_size, real constant_weight) {

    int a, d, ctx_ix, label;
    long long ctx_offset, c;
    real dot_product, outer_gradient;

    for (a = window_start; a < window_end; a++) {
        
        // skip the target word
        if (a == target) continue;
        // if this context word is unknown, ignore it and move on
        if (context_window[a] < 0) continue;
        
        // otherwise, calculate gradients from
        // (i) this positive (observed) context
        // (ii) randomly-selected negative (unobserved) contexts
        for (d = 0; d < negative + 1; d++) {
            // use observed context word as positive target
            if (d == negative) {
                ctx_ix = context_window[a];
                dot_product = pos_ctx_dots[pos_ctx_start_ix + a];
                label = 1;
            // grab a pre-selected negative target
            } else {
                ctx_ix = negative_samples[ns_start_ix + (a * negative) + d];
                dot_product = neg_ctx_dots[neg_ctx_start_ix + (a * negative) + d];
                label = 0;
            }

            ctx_offset = ctx_ix * embedding_size;

            // get the outer sigmoid gradient for this (target->ctx) pairing
            outer_gradient = CalculateLogSigmoidOuterGradient(dot_product, label);

            // add the context-based error signal for the target word
            for (c = 0; c < embedding_size; c++) {
                gradients[gradient_start_ix + c] +=
                    outer_gradient * ctx_embeddings[c + ctx_offset] * constant_weight;
            }
            // and add the target-word-based error signal for the context embeddings
            if (label == 1) {
                for (c = 0; c < embedding_size; c++) {
                    pos_ctx_gradients[pos_ctx_gradient_start_ix + (a*embedding_size) + c]
                        += outer_gradient * embeddings[c + trg_offset] * constant_weight;
                }
            } else {
                for (c = 0; c < embedding_size; c++) {
                    neg_ctx_gradients[neg_ctx_gradient_start_ix + (((a * negative) + d)*embedding_size) + c]
                        += outer_gradient * embeddings[c + trg_offset] * constant_weight;
                }
            }

        }
    }

    #ifdef PRINTGRADIENTS
    for (c = 0; c < embedding_size; c++)
        fprintf(stderr, "%f ", gradients[gradient_start_ix + c]);
    #endif
}



/**
 * Apply the gradients from a single training window to all
 * model components
 *  1. Word embeddings
 *  2. Term embeddings
 *  3. Entity embeddings
 *  4. Context embeddings
 *  5. Term-entity likelihoods
 */
void GradientAscent(int word_ix, long long word_offset, real *word_embeddings,
        real *word_gradient, real *member_word_gradients, int ttl_num_member_words,
        real *word_norms, int num_completed_terms,
        int *term_ixes, long long *term_offsets, real *term_embeddings,
        real *term_gradients, real *term_norms, int *entities_per_term,
        int *entity_ixes, long long *entity_offsets, real *entity_embeddings,
        real *entity_gradients, real *entity_norms, int *masked_word_context_window,
        int *word_negative_samples, struct term_annotation **completed_term_buffer,
        int *term_negative_samples, real *ctx_embeddings,
        real *ctx_norms, real *word_pos_ctx_gradients, real *word_neg_ctx_gradients,
        real *term_pos_ctx_gradients, real *term_neg_ctx_gradients,
        real *ctx_reg_gradients, int *all_ctx_ixes, int num_ctx,
        real *local_term_entity_likelihoods,
        struct vocabulary *wv, int full_window_size, int target,
        int sub_window_skip, int negative, int max_num_entities, real alpha,
        long long embedding_size, bool word_burn, struct model_flags *flags) {

    long long c;
    int ctx_ix;
    long long ctx_offset;
    int i, j, a, d;

    long long term_gradient_offset, term_entity_block_start, word_ctx_block_start,
        term_ctx_block_start, word_ctx_gradient_offset, term_ctx_gradient_offset,
        entity_gradient_offset;

    // apply word embedding gradients
    if (!flags->disable_words && word_ix >= 0) {
        for (c = 0; c < embedding_size; c++)
            word_embeddings[word_offset + c] += (word_gradient[c] * alpha);
        //word_norms[word_ix] = Norm(word_embeddings, word_offset, embedding_size);
        /*
        if (word_norms[word_ix] >= NORM_LIMIT) {
            error("   WORD NORM BROKE FIRST: %f\n", word_norms[word_ix]);
            exit(1);
        }
        */
    }

    if (!word_burn) {

        // apply term embedding gradients
        if (!flags->disable_terms) {
            for (i = 0; i < num_completed_terms; i++) {
                if (term_ixes[i] >= 0) {
                    term_gradient_offset = i * embedding_size;

                    for (c = 0; c < embedding_size; c++)
                        term_embeddings[term_offsets[i] + c] += (term_gradients[term_gradient_offset + c] * alpha);
                    term_norms[term_ixes[i]] = Norm(term_embeddings, term_offsets[i], embedding_size);
                    if (term_norms[term_ixes[i]] >= NORM_LIMIT) {
                        error("   TERM NORM BROKE FIRST: %f\n", term_norms[term_ixes[i]]);
                        exit(1);
                    }
                }
            }
        }

        // apply entity embedding gradients
        if (!flags->disable_entities) {
            for (i = 0; i < num_completed_terms; i++) {
                if (term_ixes[i] >= 0) {
                    term_entity_block_start = i * max_num_entities;

                    for (j = 0; j < entities_per_term[i]; j++) {
                        if (entity_ixes[term_entity_block_start +j] >= 0) {
                            entity_gradient_offset = (term_entity_block_start + j) * embedding_size;
                            for (c = 0; c < embedding_size; c++)
                                entity_embeddings[entity_offsets[term_entity_block_start + j] + c] +=
                                    (entity_gradients[entity_gradient_offset + c] * alpha);
                            entity_norms[entity_ixes[term_entity_block_start + j]] =
                                Norm(entity_embeddings, entity_offsets[term_entity_block_start + j], embedding_size);
                            /*
                            if (entity_norms[entity_ixes[term_entity_block_start + j]] >= NORM_LIMIT) {
                                error("   ENTITY NORM BROKE FIRST: %f\n", entity_norms[entity_ixes[term_entity_block_start + j]]);
                                exit(1);
                            }
                            */
                        }
                    }
                }
            }
        }
    }

    // apply context embedding gradients (from use as contexts)
    for (a = sub_window_skip; a < full_window_size - sub_window_skip; a++) {
        if (a == target) continue;

        // apply gradients from words
        if (!flags->disable_words && word_ix >= 0) {
            // positive sample
            word_ctx_gradient_offset = a * embedding_size;
            ctx_ix = masked_word_context_window[a];
            if (ctx_ix >= 0) {  // only update from non-filler contexts
                ctx_offset = ctx_ix * embedding_size;
                for (c = 0; c < embedding_size; c++)
                    ctx_embeddings[ctx_offset + c]
                        += (word_pos_ctx_gradients[word_ctx_gradient_offset + c] * alpha);
            }

            // apply gradients from negative samples
            for (d = 0; d < negative; d++) {
                word_ctx_block_start = a * negative;
                word_ctx_gradient_offset = (word_ctx_block_start + d) * embedding_size;
                ctx_ix = word_negative_samples[word_ctx_block_start + d];
                if (ctx_ix >= 0) {  // only update from non-filler negative samples
                    ctx_offset = ctx_ix * embedding_size;
                    for (c = 0; c < embedding_size; c++)
                        ctx_embeddings[ctx_offset + c]
                            += (word_neg_ctx_gradients[word_ctx_gradient_offset + c] * alpha);
                }
            }
        }

        if (!word_burn && (!flags->disable_terms || !flags->disable_entities)) {
            // apply gradients from terms and entities
            for (i = 0; i < num_completed_terms; i++) {
                // make sure it's a valid term
                if (term_ixes[i] >= 0 && completed_term_buffer[i]->contexts != NULL) {
                    // positive sample
                    term_ctx_block_start = (i * full_window_size) + a;
                    term_ctx_gradient_offset = term_ctx_block_start * embedding_size;
                    ctx_ix = completed_term_buffer[i]->contexts[a];
                    if (ctx_ix >= 0 && ctx_ix < wv->vocab_size) {  // only update from non-filler contexts
                        ctx_offset = ctx_ix * embedding_size;
                        for (c = 0; c < embedding_size; c++)
                            ctx_embeddings[ctx_offset + c]
                                += (term_pos_ctx_gradients[term_ctx_gradient_offset + c] * alpha);
                    }

                    // apply gradients from negative samples
                    for (d = 0; d < negative; d++) {
                        term_ctx_block_start = ((i * full_window_size) + a) * negative;
                        term_ctx_gradient_offset = (term_ctx_block_start + d) * embedding_size;
                        ctx_ix = term_negative_samples[term_ctx_block_start + d];
                        if (ctx_ix >= 0 && ctx_ix < wv->vocab_size) {  // only update from non-filler negative samples
                            ctx_offset = ctx_ix * embedding_size;
                            for (c = 0; c < embedding_size; c++)
                                ctx_embeddings[ctx_offset + c]
                                    += (term_neg_ctx_gradients[term_ctx_gradient_offset + c] * alpha);
                        }
                    }
                }
            }
        }
    }
    // finally, update applicable norms for context embeddings
    for (i = 0; i < num_ctx; i++) {
        if (all_ctx_ixes[i] >= 0) {
            ctx_norms[all_ctx_ixes[i]] = 
                Norm(ctx_embeddings, all_ctx_ixes[i] * embedding_size, embedding_size);
        }
    }
}


/**
 * Update model parameters for one training step.  Consists of three phases:
 *   1. Calculate context-sensitive posteriors for:
 *      a) entity likelihood
 *   2. Calculate gradients, based on:
 *      a) positive/negative contexts
 *   3. Apply batched gradients
 */
void LearningStep(int *masked_word_context_window, int target, int full_window_size,
        int sub_window_skip, struct term_annotation **completed_term_buffer, int num_completed_terms,
        int *sampled_completed_term_ixes, int *word_negative_samples, int *term_negative_samples,
        struct vocabulary *wv, struct vocabulary *tv, struct vocabulary *ev, struct entity_map *termmap,
        struct term_monogamy_map *monomap,
        int max_num_entities, real *word_embeddings, real *term_embeddings, real *entity_embeddings,
        real *ctx_embeddings, real *word_norms, real *term_norms, real *entity_norms, real *ctx_norms,
        int *entity_update_counters, int *ctx_update_counters,
        real alpha, long long embedding_size, int negative, bool word_burn,
        struct model_flags *flags) {

    int i, j, k, l, ctr, ix;
    int term_map_ix, num_entities;
    long long c;
    long long term_block_start, term_entity_block_start, term_gradient_offset,
         term_ns_block_start, entity_gradient_offset, entity_block_start,
         entity_ns_block_start;

    int word_ix = -1;
    int term_ixes[num_completed_terms];
    int entity_ixes[num_completed_terms * max_num_entities], entity_ix;

    int entities_per_term[num_completed_terms];

    long long word_offset = -1;
    long long term_offsets[num_completed_terms];
    long long entity_offsets[num_completed_terms * max_num_entities];

    real *local_term_entity_likelihoods = NULL;
    if (!flags->disable_terms || !flags->disable_entities) {
        local_term_entity_likelihoods = MallocOrDie(num_completed_terms * max_num_entities * sizeof(real), "local term entity likelihoods");
    }

    real *word_gradient = NULL, *term_gradients = NULL, *entity_gradients = NULL;
    if (!flags->disable_words) word_gradient = MallocOrDie(embedding_size * sizeof(real), "word gradient");
    if (!flags->disable_terms) term_gradients = MallocOrDie(num_completed_terms * embedding_size * sizeof(real), "term gradients");
    if (!flags->disable_entities) entity_gradients = MallocOrDie(num_completed_terms * max_num_entities * embedding_size * sizeof(real), "entity gradients");

    real *word_pos_ctx_dots = NULL, *word_neg_ctx_dots = NULL,
         *term_pos_ctx_dots = NULL, *term_neg_ctx_dots = NULL,
         *entity_pos_ctx_dots = NULL, *entity_neg_ctx_dots = NULL;
    if (!flags->disable_words) {
        word_pos_ctx_dots = MallocOrDie(full_window_size * sizeof(real), "word pos ctx dots");
        word_neg_ctx_dots = MallocOrDie(full_window_size * negative * sizeof(real), "word neg ctx dots");
    }
    if (!flags->disable_terms) {
        term_pos_ctx_dots = MallocOrDie(num_completed_terms * full_window_size * sizeof(real), "term pos ctx dots");
        term_neg_ctx_dots = MallocOrDie(num_completed_terms * full_window_size * negative * sizeof(real), "term neg ctx dots");
    }
    if (!flags->disable_entities) {
        entity_pos_ctx_dots = MallocOrDie(num_completed_terms * max_num_entities * full_window_size * sizeof(real), "entity pos ctx dots");
        entity_neg_ctx_dots = MallocOrDie(num_completed_terms * max_num_entities * full_window_size * negative * sizeof(real), "entity neg ctx dots");
    }

    real *combined_word_embeddings = MallocOrDie(num_completed_terms * embedding_size * sizeof(real), "combined word embeddings");
    real *averaged_ctx_embeddings = MallocOrDie(num_completed_terms * embedding_size * sizeof(real), "averaged ctx embeddings");

    real *word_pos_ctx_gradients = NULL, *word_neg_ctx_gradients = NULL,
         *term_pos_ctx_gradients = NULL, *term_neg_ctx_gradients = NULL;
    if (!flags->disable_words) {
        word_pos_ctx_gradients = MallocOrDie(full_window_size * embedding_size * sizeof(real), "word pos ctx gradients");
        word_neg_ctx_gradients = MallocOrDie(full_window_size * negative * embedding_size * sizeof(real), "word neg ctx gradients");
    }
    if (!flags->disable_terms || !flags->disable_entities) {
        term_pos_ctx_gradients = MallocOrDie(num_completed_terms * full_window_size * embedding_size * sizeof(real), "term pos ctx gradients");
        term_neg_ctx_gradients = MallocOrDie(num_completed_terms * full_window_size * negative * embedding_size * sizeof(real), "term neg ctx gradients");
    }

    int num_ctx = (1 + num_completed_terms * (max_num_entities + 1)) * full_window_size * (negative + 1);
    int *all_ctx_ixes = MallocOrDie(num_ctx * sizeof(int), "all ctx ixes");
    real *ctx_reg_gradients = MallocOrDie(num_ctx * embedding_size * sizeof(real), "ctx reg gradients");

    int ttl_num_member_words = 0;
    for (i = 0; i < num_completed_terms; i++)
        ttl_num_member_words += completed_term_buffer[i]->num_tokens;
    real *member_word_gradients = MallocOrDie(ttl_num_member_words * embedding_size * sizeof(real), "member word gradients");



    ////////////////////////////////////////
    // Set up for gradient calculation
    ////////////////////////////////////////

    // calculate embedding start locations for all relevant words...
    if (!flags->disable_words) {
        word_ix = masked_word_context_window[target];
        if (word_ix >= 0) word_offset = word_ix * embedding_size;
        else word_offset = -1;
    }
    // ...terms...
    if (!flags->disable_terms) {
        for (i = 0; i < num_completed_terms; i++) {
            term_ixes[i] = sampled_completed_term_ixes[i];
            // calculate term offset
            if (term_ixes[i] >= 0) term_offsets[i] = term_ixes[i] * embedding_size;
            else term_offsets[i] = -1;
        }
    }
    // ...and entities
    if (!flags->disable_entities) {
        for (i = 0; i < num_completed_terms * max_num_entities; i++) entity_ixes[i] = -1;  // default value
        for (i = 0; i < num_completed_terms; i++) {
            ix = term_ixes[i];
            entities_per_term[i] = 0;

            // ignore entities for downsampled terms
            if (ix >= 0) {
                term_map_ix = SearchMap(termmap, &tv->vocab[ix]);
                if (term_map_ix == -1) {
                    error(">>> Encountered unmapped term %s\n", &tv->vocab[ix].word);
                    continue;
                }

                // save the indices and calculate offsets for these entities
                num_entities = termmap->map[term_map_ix].num_entities;
                ctr = 0;
                for (j = 0; j < num_entities; j++) {
                    ix = termmap->map[term_map_ix].entities[j].vocab_index;
                    term_entity_block_start = i * max_num_entities;
                    entity_ixes[term_entity_block_start + j] = ix;
                    if (ix >= 0) entity_offsets[term_entity_block_start + j] = entity_ixes[term_entity_block_start + j] * embedding_size;
                    else entity_offsets[term_entity_block_start + j] = -1;
                    ctr++;
                }

                entities_per_term[i] = num_entities;
            }
        }
    }


    // grab the indices of all the context words we'll use
    TrackContextIndices(masked_word_context_window, word_ix,
        word_negative_samples, completed_term_buffer, term_negative_samples,
        num_completed_terms, sampled_completed_term_ixes, sub_window_skip,
        full_window_size - sub_window_skip, full_window_size, target, negative,
        all_ctx_ixes, num_ctx, flags);



    ////////////////////////////////////////////////////////////
    // Calculate dot products with positive/negative contexts
    ////////////////////////////////////////////////////////////
    
    // words
    if (!flags->disable_words && word_ix >= 0) {
        CalculateContextDotProducts(word_embeddings, word_offset, ctx_embeddings,
            masked_word_context_window, word_negative_samples, negative, 0,
            full_window_size, target, sub_window_skip, full_window_size - sub_window_skip,
            embedding_size, word_pos_ctx_dots, 0, word_neg_ctx_dots, 0);
    }
    // terms
    if (!word_burn && (!flags->disable_terms || !flags->disable_entities)) {
        for (i = 0; i < num_completed_terms; i++) {
            if (term_ixes[i] >= 0) {
                term_block_start = i * full_window_size;
                term_ns_block_start = i * full_window_size * negative;

                if (!flags->disable_terms) {
                    CalculateContextDotProducts(term_embeddings, term_offsets[i],
                        ctx_embeddings, completed_term_buffer[i]->contexts,
                        term_negative_samples, negative, term_ns_block_start,
                        full_window_size, target, sub_window_skip, full_window_size - sub_window_skip,
                        embedding_size, term_pos_ctx_dots, term_block_start, term_neg_ctx_dots,
                        term_ns_block_start);
                }


        //      entities
                if (!flags->disable_entities) {
                    term_entity_block_start = i * max_num_entities;
                    for (j = 0; j < entities_per_term[i]; j++) {
                        if (entity_ixes[term_entity_block_start + j] >= 0) {
                            entity_block_start = ((i * max_num_entities) + j) * full_window_size;
                            entity_ns_block_start = ((i * max_num_entities) + j) * full_window_size * negative;

                            if (completed_term_buffer[i]->contexts == NULL) {
                                error("  [WARNING] entity with unallocated contexts\n");
                                for (k = 0; k < full_window_size; k++) {
                                    entity_pos_ctx_dots[entity_block_start+k] = 0;
                                    for (l = 0; l < negative; l++) {
                                        entity_pos_ctx_dots[entity_ns_block_start+(k*negative)+l] = 0;
                                    }
                                }
                            } else {
                                CalculateContextDotProducts(entity_embeddings,
                                    entity_offsets[term_entity_block_start + j], ctx_embeddings,
                                    completed_term_buffer[i]->contexts,
                                    term_negative_samples, negative, term_ns_block_start,
                                    full_window_size, target, sub_window_skip,
                                    full_window_size - sub_window_skip, embedding_size,
                                    entity_pos_ctx_dots, entity_block_start,
                                    entity_neg_ctx_dots, entity_ns_block_start);
                            }
                        }
                    }
                }
            }
        }
    }


    ////////////////////////////////////////////////////////////
    // Calculate all other multi-use info
    ////////////////////////////////////////////////////////////

    if (!word_burn && (!flags->disable_terms || !flags->disable_entities)) {
        for (i = 0; i < num_completed_terms; i++) {
            term_entity_block_start = i * max_num_entities;
            for (j = 0; j < entities_per_term[i]; j++)
                local_term_entity_likelihoods[term_entity_block_start + j] = 1/(real)entities_per_term[i];
            for (j = entities_per_term[i]; j < max_num_entities; j++)
                local_term_entity_likelihoods[term_entity_block_start + j] = 0;
        }
    }


    /////////////////////////////
    // Calculate gradients
    /////////////////////////////

    // (1) zero out all gradient arrays
    if (!flags->disable_words) {
        for (c = 0; c < embedding_size; c++)
            word_gradient[c] = 0;
        for (c = 0; c < ttl_num_member_words * embedding_size; c++)
            member_word_gradients[c] = 0;
        for (c = 0; c < full_window_size * embedding_size; c++)
            word_pos_ctx_gradients[c] = 0;
        for (c = 0; c < full_window_size * negative * embedding_size; c++)
            word_neg_ctx_gradients[c] = 0;
    }
    if (!flags->disable_terms) {
        for (c = 0; c < num_completed_terms * embedding_size; c++)
            term_gradients[c] = 0;
    }
    if (!flags->disable_entities) {
        for (c = 0; c < num_completed_terms * max_num_entities * embedding_size; c++)
            entity_gradients[c] = 0;
    }
    if (!flags->disable_terms || !flags->disable_entities) {
        for (c = 0; c < num_completed_terms * full_window_size * embedding_size; c++)
            term_pos_ctx_gradients[c] = 0;
        for (c = 0; c < num_completed_terms * full_window_size * negative * embedding_size; c++)
            term_neg_ctx_gradients[c] = 0;
    }

    // (1) Word gradients
    if (!flags->disable_words && word_ix >= 0) {
        
        // mini-batched word gradients from contexts
        #ifdef PRINTGRADIENTS
        fprintf(stderr, "Word context based gradient: [ ");
        #endif
        AddContextBasedGradients(word_embeddings, word_offset,
            ctx_embeddings, masked_word_context_window, word_negative_samples, negative, 0,
            full_window_size, target, sub_window_skip, full_window_size - sub_window_skip,
            word_pos_ctx_dots, 0, word_neg_ctx_dots, 0,
            word_gradient, 0, word_pos_ctx_gradients, 0, word_neg_ctx_gradients, 0, embedding_size, 1);
        #ifdef PRINTGRADIENTS
        fprintf(stderr, "]\n");
        fflush(stderr);
        #endif
    }

    if (!word_burn && (!flags->disable_terms || !flags->disable_entities)) {
        // (2-3) Term/entity gradients
        for (i = 0; i < num_completed_terms; i++) {
            if (term_ixes[i] >= 0) {
                term_block_start = i * full_window_size;
                term_ns_block_start = i * full_window_size * negative;
                term_gradient_offset = i * embedding_size;
                
                // mini-batched term embedding gradients from contexts
                if (!flags->disable_terms) {
                    #ifdef PRINTGRADIENTS
                    fprintf(stderr, "Term context based gradient: [ ");
                    #endif
                    AddContextBasedGradients(term_embeddings, term_offsets[i],
                        ctx_embeddings, completed_term_buffer[i]->contexts,
                        term_negative_samples, negative, term_ns_block_start,
                        full_window_size, target, sub_window_skip, full_window_size - sub_window_skip,
                        term_pos_ctx_dots, term_block_start, term_neg_ctx_dots, term_ns_block_start,
                        term_gradients, term_gradient_offset, term_pos_ctx_gradients,
                        i * full_window_size * embedding_size, term_neg_ctx_gradients,
                        term_ns_block_start * embedding_size, embedding_size, 1);
                    #ifdef PRINTGRADIENTS
                    fprintf(stderr, "]\n");
                    fflush(stderr);
                    #endif
                }
                
                // (3) Entity gradients
                if (!flags->disable_entities) {
                    term_entity_block_start = i * max_num_entities;
                    term_ns_block_start = i * full_window_size * negative;
                    for (j = 0; j < entities_per_term[i]; j++) {
                        if (entity_ixes[term_entity_block_start + j] >= 0) {
                            entity_block_start = ((i * max_num_entities) + j) * full_window_size;
                            entity_ns_block_start = ((i * max_num_entities) + j) * full_window_size * negative;
                            entity_gradient_offset = (term_entity_block_start + j) * embedding_size;

                            // mini-batched entity gradients from contexts
                            if (completed_term_buffer[i]->contexts != NULL) {
                                #ifdef PRINTGRADIENTS
                                fprintf(stderr, "Entity context based gradient: [ ");
                                #endif
                                AddContextBasedGradients(entity_embeddings, entity_offsets[term_entity_block_start + j],
                                    ctx_embeddings, completed_term_buffer[i]->contexts,
                                    term_negative_samples, negative, term_ns_block_start,
                                    full_window_size, target, sub_window_skip, full_window_size - sub_window_skip,
                                    entity_pos_ctx_dots, entity_block_start, entity_neg_ctx_dots, entity_ns_block_start,
                                    entity_gradients, entity_gradient_offset,
                                    term_pos_ctx_gradients, i * full_window_size * embedding_size,
                                    term_neg_ctx_gradients, term_ns_block_start * embedding_size, embedding_size,
                                    local_term_entity_likelihoods[term_entity_block_start + j]);
                                #ifdef PRINTGRADIENTS
                                fprintf(stderr, "]\n");
                                fflush(stderr);
                                #endif
                            }
                        }
                    }
                }
            }
        }
    }


    /////////////////////////////
    // Take gradient ascent step
    /////////////////////////////
    
    GradientAscent(word_ix, word_offset, word_embeddings, word_gradient, member_word_gradients,
        ttl_num_member_words, word_norms, num_completed_terms, term_ixes, term_offsets,
        term_embeddings, term_gradients, term_norms, entities_per_term, entity_ixes,
        entity_offsets, entity_embeddings, entity_gradients, entity_norms,
        masked_word_context_window, word_negative_samples, completed_term_buffer,
        term_negative_samples, ctx_embeddings, ctx_norms, word_pos_ctx_gradients,
        word_neg_ctx_gradients, term_pos_ctx_gradients, term_neg_ctx_gradients, ctx_reg_gradients,
        all_ctx_ixes, num_ctx, local_term_entity_likelihoods, wv, full_window_size, target,
        sub_window_skip, negative, max_num_entities, alpha, embedding_size, word_burn,
        flags);


    /////////////////////////////
    // Reset counters
    /////////////////////////////

    // entities
    if (!flags->disable_entities) {
        for (i = 0; i < num_completed_terms; i++) {
            if (term_ixes[i] >= 0) {
                for (j = 0; j < entities_per_term[i]; j++) {
                    entity_ix = entity_ixes[(i*max_num_entities)+j];
                    if (entity_ix >= 0)
                        entity_update_counters[entity_ix] = 0;
                }
            }
        }
    }
    // contexts
    for (i = 0; i < num_ctx; i++) {
        if (all_ctx_ixes[i] >= 0)
            ctx_update_counters[all_ctx_ixes[i]] = 0;
    }

    // FreeAndNull checks for if NULL before freeing,
    // so can safely call on everything here
    FreeAndNull((void *)&local_term_entity_likelihoods);

    FreeAndNull((void *)&word_gradient);
    FreeAndNull((void *)&term_gradients);
    FreeAndNull((void *)&entity_gradients);

    FreeAndNull((void *)&word_pos_ctx_dots);
    FreeAndNull((void *)&word_neg_ctx_dots);
    FreeAndNull((void *)&term_pos_ctx_dots);
    FreeAndNull((void *)&term_neg_ctx_dots);
    FreeAndNull((void *)&entity_pos_ctx_dots);
    FreeAndNull((void *)&entity_neg_ctx_dots);
    
    FreeAndNull((void *)&combined_word_embeddings);
    FreeAndNull((void *)&averaged_ctx_embeddings);

    FreeAndNull((void *)&word_pos_ctx_gradients);
    FreeAndNull((void *)&word_neg_ctx_gradients);
    FreeAndNull((void *)&term_pos_ctx_gradients);
    FreeAndNull((void *)&term_neg_ctx_gradients);

    FreeAndNull((void *)&all_ctx_ixes);
    FreeAndNull((void *)&ctx_reg_gradients);

    FreeAndNull((void *)&member_word_gradients);

}
