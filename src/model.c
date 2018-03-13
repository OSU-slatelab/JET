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
    (*flags)->disable_compositionality = false;
    (*flags)->disable_likelihoods = false;
    (*flags)->disable_term_similarity = false;
    (*flags)->disable_latency = false;
    (*flags)->disable_regularization = false;
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
 * Initialize term compositionality scores (range [-1,1]);
 * by default, all start at 0.
 */
void InitCompositionalityScores(real **global_term_compositionality_scores, struct vocabulary *tv) {
    *global_term_compositionality_scores = malloc(tv->vocab_size * sizeof(real));
    if (*global_term_compositionality_scores == NULL) {
        error("   >>> Failed to allocate memory for term compositionality scores; Aborting\n");
        exit(1);
    }
    for (int i = 0; i < tv->vocab_size; i++) {
        (*global_term_compositionality_scores)[i] = 0;
    }
}
/**
 * Destroy term compositionality scores
 */
void DestroyCompositionalityScores(real **global_term_compositionality_scores) {
    if (*global_term_compositionality_scores != NULL) {
        free(*global_term_compositionality_scores);
        *global_term_compositionality_scores = NULL;
    }
}

/**
 * Initialize prior probability distribution over possible concepts for each term.
 * Term-entity likelihoods are aligned to vocabulary indices; each term is given
 * MaxNumEntities(em) slots for likelihoods.
 * 
 * For now, initializes to uniform distribution.
 * TODO: introduce ranking-based initialization
 */
void InitTermEntityLikelihoods(real **global_term_entity_likelihoods, struct vocabulary *tv, 
        struct entity_map *em) {
    long i, j;
    int map_ix, num_entities;

    int max_num_entities = MaxNumEntities(em);
    *global_term_entity_likelihoods = calloc(tv->vocab_size * max_num_entities, sizeof(real));
    if (*global_term_entity_likelihoods == NULL) {
        error("   >>> Failed to allocate memory for term->entity likelihoods; Aborting\n");
        exit(1);
    }
    // for each term, initialize the prior distribution over its possible
    // concepts to uniform
    for (i = 0; i < tv->vocab_size; i++) {
        map_ix = SearchMap(em, &tv->vocab[i]);
        if (map_ix >= 0) {
            num_entities = em->map[map_ix].num_entities;
            for (j = 0; j < num_entities; j++) {
                (*global_term_entity_likelihoods)[(i*max_num_entities)+j]
                    = 1.0/num_entities;
            }
        }
    }
}
/**
 * Destroy term->entity likelihoods
 */
void DestroyTermEntityLikelihoods(real **global_term_entity_likelihoods) {
    if (*global_term_entity_likelihoods != NULL) {
        free(*global_term_entity_likelihoods);
        *global_term_entity_likelihoods = NULL;
    }
}

/**
 * Initialize interpolation weights for terms and averaged ctx
 * to 0.5 across the board (i.e., start with averaging)
 */
void InitInterpolationWeights(real **term_transform_weights, real **ctx_transform_weights,
        long long embedding_size) {
    long long c;

    *term_transform_weights = malloc(embedding_size * sizeof(real));
    if (*term_transform_weights == NULL) {
        error("   >>> Failed to allocate memory for term interpolation weights; Aborting\n");
        exit(1);
    }
    for (c = 0; c < embedding_size; c++) {
        (*term_transform_weights)[c] = 0.5;
    }

    *ctx_transform_weights = malloc(embedding_size * sizeof(real));
    if (*ctx_transform_weights == NULL) {
        error("   >>> Failed to allocate memory for ctx interpolation weights; Aborting\n");
        exit(1);
    }
    for (c = 0; c < embedding_size; c++) {
        (*ctx_transform_weights)[c] = 0.5;
    }
}

/**
 * Destroy interpolation weights
 */
void DestroyInterpolationWeights(real **term_transform_weights, real **ctx_transform_weights) {
    if (*term_transform_weights != NULL) {
        free(*term_transform_weights);
        *term_transform_weights = NULL;
    }
    if (*ctx_transform_weights != NULL) {
        free(*ctx_transform_weights);
        *ctx_transform_weights = NULL;
    }
}

/**
 * Wrapper for all model initialization steps
 */
void InitializeModel(real **word_embeddings, real **term_embeddings, real **entity_embeddings,
        real **ctx_embeddings, real **word_norms, real **term_norms, real **entity_norms,
        real **ctx_norms, real **term_transform_weights, real **ctx_transform_weights,
        real **global_term_compositionality_scores, real **global_term_entity_likelihoods,
        struct vocabulary *wv, struct vocabulary *tv, struct vocabulary *ev, struct entity_map *em,
        long long embedding_size, int **unitable, real **word_downsampling_table,
        real **term_downsampling_table, real downsampling_rate) {
    InitExpTable();
    InitCompositionalityScores(global_term_compositionality_scores, tv);
    InitTermEntityLikelihoods(global_term_entity_likelihoods, tv, em);
    InitInterpolationWeights(term_transform_weights, ctx_transform_weights, embedding_size);
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
        real **ctx_norms, real **term_transform_weights, real **ctx_transform_weights,
        real **global_term_compositionality_scores, real **global_term_entity_likelihoods,
        int **unitable, real **word_downsampling_table, real **term_downsampling_table) {
    DestroyExpTable();
    DestroyCompositionalityScores(global_term_compositionality_scores);
    DestroyTermEntityLikelihoods(global_term_entity_likelihoods);
    DestroyInterpolationWeights(term_transform_weights, ctx_transform_weights);
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
 * Given a set of term mentions with invdividual context windows,
 * calculate a context-sensitive posterior probability of each
 * term having a latent reference (to an entity).
 *
 * Let E be the entities that term t can represent, and let W
 * be the observed context window, where Avg(V) is its average
 * word embedding.
 * Then, this is calculated as:
 *   t_lat <- max_{e \in E} CosSim(v_e, AvgV(W))
 */
void CalculateLocalTermLatencyScores(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, int max_num_entities,
        int *entities_per_term, real *entity_embeddings, real *entity_norms,
        int *entity_ixes, real *averaged_ctx_embeddings, long long embedding_size,
        real *local_term_latency_scores) {
    int i, j, entity_ix, term_entity_block_start;
    real entity_similarity, max_entity_similarity;
    
    // zero it out first
    for (i = 0; i < num_completed_terms; i++)
        local_term_latency_scores[i] = 0;

    for (i = 0; i < num_completed_terms; i++) {
        if (sampled_completed_term_ixes[i] >= 0) {
            term_entity_block_start = i * max_num_entities;

            // find the maximum cosine similarity between each of this
            // term's possible entities and the averaged context window
            max_entity_similarity = 0;
            for (j = 0; j < entities_per_term[i]; j++) {
                entity_ix = entity_ixes[term_entity_block_start + j];

                entity_similarity = CosineSimilarity(entity_embeddings,
                    entity_ix * embedding_size, averaged_ctx_embeddings,
                    i * embedding_size, entity_norms[entity_ix],
                    Norm(averaged_ctx_embeddings, i*embedding_size, embedding_size),
                    embedding_size);

                if (entity_similarity > max_entity_similarity)
                    max_entity_similarity = entity_similarity;
            }

            // floor it at 0.1
            if (max_entity_similarity < 0.1)
                max_entity_similarity = 0.1;

            // and save it
            local_term_latency_scores[i] = max_entity_similarity;
        }
    }
}

/**
 * Given a set of term mentions with individual surface forms,
 * calculate a context-dependent compositionality score
 * for each term.
 *
 * For term t with member words W, this is calculated as:
 *   comp_t <- cos(e_t, \sum_{w \in W} e_w)
 */
/*
void CalculateLocalTermCompositionalityScores(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, long long embedding_size,
        real *term_embeddings, real *term_norms, real *averaged_word_embeddings,
        real *local_term_compositionality_scores) {
    long long word_avg_offset, term_offset;
    int i, term_ix;
    real cos_sim;

    for (i = 0; i < num_completed_terms; i++) {
        if (sampled_completed_term_ixes[i] >= 0) {
            term_ix = sampled_completed_term_ixes[i];
            term_offset = term_ix * embedding_size;
            word_avg_offset = i * embedding_size;

            // calculate cosine similarity
            cos_sim = CosineSimilarity(term_embeddings, term_offset,
                averaged_word_embeddings, word_avg_offset, term_norms[term_ix],
                Norm(averaged_word_embeddings, word_avg_offset, embedding_size),
                embedding_size);

            // and store in the compositionality posteriors
            local_term_compositionality_scores[i] = cos_sim;
        }
    }
}
*/


/**
 * Given a set of term mentions with individual context windows,
 * calculate a context-sensitive probability distribution over the
 * entities each term can represent.
 *
 * This distribution also takes into account the current
 * context-independent term-entity likelihoods.
 *
 * For term t, representing entities E, * let pr(e \in E |t)
 * be the context-independent prior of e given * t.  Then, with
 * context window C, posteriors are calculated as:
 *
 *   score(e \in E, t, C) <- pr(e|t) * ( [\sum_{c \in C} e . c] / norm(c) )
 *   Z <- (1/|E|) * \sum_{e \in E} score(e, C)
 *
 *   P(e \in E | t, C) <- score(e,t,C) / Z
 */
void CalculateLocalTermEntityLikelihoods(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, int max_num_entities,
        int *entities_per_term, real *global_term_entity_likelihoods, real *entity_embeddings,
        real *entity_norms, int *entity_ixes, long long *entity_offsets,
        real *entity_pos_ctx_dots, real *ctx_norms, int full_window_size, int target,
        long long embedding_size, real *local_term_entity_likelihoods) {
    int i, j, term_ix, window_ix, ctx_ix;
    long long term_entity_block_start, term_entity_likelihood_start, entity_block_start;
    real cos_sim, normalization_term;

    // zero it out first
    for (i = 0; i < num_completed_terms * max_num_entities; i++)
        local_term_entity_likelihoods[i] = 0;

    for (i = 0; i < num_completed_terms; i++) {
        term_ix = sampled_completed_term_ixes[i];
        if (term_ix >= 0) {
            term_entity_block_start = i * max_num_entities;
            term_entity_likelihood_start = term_ix * max_num_entities;

            // for each entity, do the following steps
            for (j = 0; j < entities_per_term[i]; j++) {
                if (entity_ixes[term_entity_block_start + j] >= 0) {
                    entity_block_start = (term_entity_block_start + j) * full_window_size;

                    // (1) sum the shifted cosine similarities of each entity with every context word
                    for (window_ix = 0; window_ix < full_window_size; window_ix++) {
                        // ignore the target position
                        if (window_ix == target) continue;
                        // also, if the context word is unknown, ignore it
                        ctx_ix = completed_term_buffer[i]->contexts[window_ix];
                        if (ctx_ix < 0) continue;
                        // otherwise, use the pre-calculated dot products to get
                        // the cosine similarity
                        cos_sim = CosineSimilarityFromDot(
                            entity_pos_ctx_dots[entity_block_start + window_ix],
                            entity_norms[entity_ixes[term_entity_block_start + j]],
                            ctx_norms[ctx_ix]
                        );
                        // add 1 to force into non-negative space (for softmax calculation)
                        local_term_entity_likelihoods[term_entity_block_start + j] +=
                            (1 + cos_sim);
                    }

                    // (2) apply context-independent priors
                    local_term_entity_likelihoods[term_entity_block_start + j] *=
                        global_term_entity_likelihoods[term_entity_likelihood_start + j];
                }
            }

            // finally, take a softmax over the entities
            normalization_term = 0;
            for (j = 0; j < entities_per_term[i]; j++)
                normalization_term += local_term_entity_likelihoods[term_entity_block_start + j];
            for (j = 0; j < entities_per_term[i]; j++) {
                if (normalization_term > 0)
                    local_term_entity_likelihoods[term_entity_block_start + j] /= normalization_term;
                // default to uniform distribution if no local scores
                else
                    local_term_entity_likelihoods[term_entity_block_start + j] = 1.0/entities_per_term[i];
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
 * Calculates term gradients based on their similarity to their member words,
 * and adds to the current gradient vector.
 *
 * Let W' be the average embedding of the member words of t, and let comp[t]
 * be the current compositionality score of t.
 * Then, the compositionality gradient is calculated as:
 *   g_t = grad(e_t . W', 1) * mono[t] * W'
 *   \forall w \in W'
 *      g_w = grad(e_t . W', 1) * mono[t] * P(t|w) * t
 */
void AddMemberWordBasedGradients(real *term_embeddings, int term_ix,
        long long term_offset, real *combined_word_embeddings,
        long long word_comb_offset, int *member_words, int num_tokens,
        struct term_monogamy_map *monomap, real *term_gradients,
        long term_gradient_start_ix, real *member_word_gradients,
        long member_word_gradient_start_ix, long long embedding_size) {
    int i, known_member_words;
    long word_gradient_start;
    long long c;
    real combined_word_norm;
    real dot_product, outer_gradient;
    real monogamy_weight, word_monogamy_weight;

    // flip the monogamy weight for downweighting the update;
    // if it's too low, just ignore these gradients entirely
    monogamy_weight = 1 - monomap->monogamies[term_ix].monogamy_weight;
    if (monogamy_weight <= 0.00001) return;

    combined_word_norm = Norm(combined_word_embeddings, word_comb_offset, embedding_size);

    // gradient is non-zero only if there was at least one known member word
    if (combined_word_norm > 0) {
        
        // dot the current term and the combined member words
        dot_product = DotProduct(term_embeddings, term_offset,
            combined_word_embeddings, word_comb_offset, embedding_size);

        // calculate the outer gradient of the sigmoid scorer
        // (pushing the term towards its weighted member words)
        outer_gradient = CalculateLogSigmoidOuterGradient(dot_product, 1);

        // add in the weighted term gradient
        #ifdef PRINTGRADIENTS
        fprintf(stderr, "Term member-word based gradient: [ ");
        #endif
        for (c = 0; c < embedding_size; c++) {
            term_gradients[term_gradient_start_ix + c] += (
                outer_gradient *
                monogamy_weight *
                combined_word_embeddings[word_comb_offset + c]
            );
            #ifdef PRINTGRADIENTS
            fprintf(stderr, "%f ",(
                outer_gradient *
                monogamy_weight *
                combined_word_embeddings[word_comb_offset + c]
            ));
            #endif
        }
        #ifdef PRINTGRADIENTS
        fprintf(stderr, "]\n");
        #endif
        /*
        for (c = 0; c < embedding_size; c++) {
            if (abs(outer_gradient*compositionality_weight*combined_word_embeddings[word_avg_offset+c]) > 1) {
                printf("HIGH TERM UPDATE: %f\n", term_gradients[term_gradient_start_ix + c]);
                printf("  Comp weight: %f\n", compositionality_weight);
                printf("  Dot product: %f\n", dot_product);
                printf("  Outer gradient: %f\n", outer_gradient);
                printf("  Weighted avg word embed: [");
                for (c = 0; c < embedding_size; c++)
                    printf("%f ", weighted_combined_word_embeddings[c]);
                printf("]\n");
                printf("  Avg word embed: [");
                for (c = 0; c < embedding_size; c++)
                    printf("%f ", combined_word_embeddings[word_avg_offset+c]);
                printf("]\n");
                printf("  Term embed: [");
                for (c = 0; c < embedding_size; c++)
                    printf("%f ", term_embeddings[term_offset+c]);
                printf("]\n");
            }
            break;
        }
        */

        // count the number of known member words
        known_member_words = 0;
        for (i = 0; i < num_tokens; i++) {
            if (member_words[i] >= 0)
                known_member_words++;
        }

        // and add in the weighted gradients for the member words
        for (i = 0; i < num_tokens; i++) {
            if (member_words[i] >= 0) {
                word_monogamy_weight = monomap->monogamies[term_ix].by_word[i];

                #ifdef PRINTGRADIENTS
                fprintf(stderr, "Word %d term-based gradient: [ ", member_words[i]);
                #endif
                word_gradient_start = member_word_gradient_start_ix + (i * embedding_size);
                for (c = 0; c < embedding_size; c++) {
                    member_word_gradients[word_gradient_start + c] += (
                        monogamy_weight *
                        outer_gradient *
                        (1/(real)known_member_words) *
                        word_monogamy_weight *
                        term_embeddings[term_offset + c]
                    );
                    /*
                    if (abs(outer_gradient*(1/(real)known_member_words)*compositionality_weight*term_embeddings[term_offset+c]) > 1) {
                        printf("HIGH WORD UPDATE: %f\n", outer_gradient*(1/(real)known_member_words)*compositionality_weight*term_embeddings[term_offset+c]);
                        printf("  Comp weight: %f\n", compositionality_weight);
                        printf("  Dot product: %f\n", dot_product);
                        printf("  Outer gradient: %f\n", outer_gradient);
                        printf("  Term embed: %f\n", term_embeddings[term_offset + c]);
                        printf("  Avg word embed[c]: %f\n", combined_word_embeddings[word_avg_offset + c]);
                    }
                    */
                    #ifdef PRINTGRADIENTS
                    fprintf(stderr, "%f ",(
                        monogamy_weight *
                        outer_gradient *
                        (1/(real)known_member_words) *
                        word_monogamy_weight *
                        term_embeddings[term_offset + c]
                    ));
                    #endif
                }
                #ifdef PRINTGRADIENTS
                fprintf(stderr, "]\n");
                #endif
            }
        }
    }
}

/**
 * Adds entity, term, ctx, and interpolation gradients based on
 * entity similarity to a * context-sensitive transformation of
 * the mentioning term.
 *
 * Transformed term is calculated as a learned interpolation of
 * the term and its averaged context embeddings:
 *   \hat{t} = (t * w_t) + (AvgCtx_t * w_C)
 *
 * Where w_t and w_C are general (non term-specific) vectors learned
 * from all such transformations.
 *
 * Entity gradients are then weighted by:
 *   (1) context-dependent likelihood of this entity
 *          -> noted P(e|t,C)
 *   (2) log sigmoid gradient of entity and the transformed term
 *          -> noted g(e . \hat{t})
 *
 * The entity gradient is calculated as
 *   g_e = P(e|t,C) * g(e . \hat{t}) * \hat{t}
 * Term gradient is:
 *   g_t = P(e|t,C) * g(e . \hat{t}) * (e * w_t)
 * For each of the |C| context words, its gradient is
 *   g_Ci = P(e|t,C) * g(e . \hat{t}) * (1/|C|) * (e * w_C)
 * The term transformation weight gradient is
 *   g_w_t = P(e|t,C) * g(e . \hat{t}) * (e * t)
 * And the ctx transformation weight gradient is
 *   g_w_C = P(e|t,C) * g(e . \hat{t}) * (e * AvgCtx_t)
 */
void AddTermEntitySimilarityBasedGradients(real *term_embeddings, long long term_offset,
        real *entity_embeddings, long long entity_offset, real *averaged_ctx_embeddings,
        long long avg_ctx_offset, int *term_pos_ctx_ixes, int window_start, int window_end,
        int target, real *term_transform_weights, real *ctx_transform_weights,
        real local_term_entity_likelihood, real *entity_gradients, long entity_gradient_start_ix,
        real *term_gradients, long term_gradient_start_ix, real *term_pos_ctx_gradients,
        long long term_pos_ctx_gradient_start_ix, real *term_transform_gradients,
        real *ctx_transform_gradients, long long embedding_size) {

    real dot_product, outer_gradient;
    real weighted_term_embedding[embedding_size];
    real weighted_ctx_embedding[embedding_size];
    real interpolated_term_ctx_embedding[embedding_size];
    long long c;
    int a, num_valid_ctx_words;
    real avg_weight;

    // use the current term and ctx transforms to calculate the interpolated
    // term transformation
    for (c = 0; c < embedding_size; c++) {
        weighted_term_embedding[c] =
            term_embeddings[term_offset + c] * term_transform_weights[c];
        weighted_ctx_embedding[c] =
            averaged_ctx_embeddings[avg_ctx_offset + c] * ctx_transform_weights[c];
        interpolated_term_ctx_embedding[c] =
            weighted_term_embedding[c] + weighted_ctx_embedding[c];
    }

    // dot the entity and interpolated term/ctx
    dot_product = DotProduct(entity_embeddings, entity_offset,
        interpolated_term_ctx_embedding, 0, embedding_size);
    // use it to get the outer gradient of the sigmoid scorer
    // (pushing the entity towards this term)
    outer_gradient = CalculateLogSigmoidOuterGradient(dot_product, 1);

    // add in weighted entity gradient
    #ifdef PRINTGRADIENTS
    fprintf(stderr, "Entity term-sim based gradient: [ ");
    #endif
    for (c = 0; c < embedding_size; c++) {
        entity_gradients[entity_gradient_start_ix + c] += (
            local_term_entity_likelihood *
            outer_gradient *
            interpolated_term_ctx_embedding[c]
        );
        #ifdef PRINTGRADIENTS
        fprintf(stderr, "%f ",(
            local_term_entity_likelihood *
            outer_gradient *
            interpolated_term_ctx_embedding[c]
        ));
        #endif
    }
    #ifdef PRINTGRADIENTS
    fprintf(stderr, "]\n");
    fflush(stderr);
    #endif
    for (c = 0; c < embedding_size; c++) {
        if (abs(local_term_entity_likelihood*outer_gradient*interpolated_term_ctx_embedding[c]) > 1) {
            printf("HIGH ENTITY UPDATE: %f\n", local_term_entity_likelihood*outer_gradient*interpolated_term_ctx_embedding[c]);
            printf("  Local probability: %f\n", local_term_entity_likelihood);
            printf("  Dot product: %f\n", dot_product);
            printf("  Outer gradient: %f\n", outer_gradient);
            printf("  Term weights: [");
            for (c = 0; c < embedding_size; c++)
                printf("%f ", term_transform_weights[c]);
            printf("]\n");
            printf("  Ctx weights: [");
            for (c = 0; c < embedding_size; c++)
                printf("%f ", ctx_transform_weights[c]);
            printf("]\n");
            printf("  Term embed: [");
            for (c = 0; c < embedding_size; c++)
                printf("%f ", term_embeddings[term_offset+c]);
            printf("]\n");
            printf("  Ctx embed: [");
            for (c = 0; c < embedding_size; c++)
                printf("%f ", averaged_ctx_embeddings[avg_ctx_offset+c]);
            printf("]\n");
            printf("  Weighted term embed: [");
            for (c = 0; c < embedding_size; c++)
                printf("%f ", weighted_term_embedding[c]);
            printf("]\n");
            printf("  Weighted ctx embed: [");
            for (c = 0; c < embedding_size; c++)
                printf("%f ", weighted_ctx_embedding[c]);
            printf("]\n");
            printf("  Interpolated embed: [");
            for (c = 0; c < embedding_size; c++)
                printf("%f ", interpolated_term_ctx_embedding[c]);
            printf("]\n");
            printf("  Entity embed: [");
            for (c = 0; c < embedding_size; c++)
                printf("%f ", entity_embeddings[entity_offset+c]);
            printf("]\n");
        }
        break;
    }

    // add in weighted term gradients
    #ifdef PRINTGRADIENTS
    fprintf(stderr, "Term entity-sim based gradient: [ ");
    #endif
    for (c = 0; c < embedding_size; c++) {
        term_gradients[term_gradient_start_ix + c] += (
            local_term_entity_likelihood *
            outer_gradient *
            term_transform_weights[c] *
            entity_embeddings[entity_offset + c]
        );
        #ifdef PRINTGRADIENTS
        fprintf(stderr, "%f ",(
            local_term_entity_likelihood *
            outer_gradient *
            term_transform_weights[c] *
            entity_embeddings[entity_offset + c]
        ));
        #endif
    }
    #ifdef PRINTGRADIENTS
    fprintf(stderr, "]\n");
    fflush(stderr);
    #endif

    // add in weighted gradients for each of the context words
    num_valid_ctx_words = 0;
    for (a = window_start; a < window_end; a++) {
        if (a == target) continue;
        if (term_pos_ctx_ixes[a] >= 0)
            num_valid_ctx_words++;
    }
    if (num_valid_ctx_words > 0) {
        avg_weight = 1/(real)num_valid_ctx_words;
        for (a = window_start; a < window_end; a++) {
            if (a == target) continue;
            if (term_pos_ctx_ixes[a] >= 0) {
                #ifdef PRINTGRADIENTS
                fprintf(stderr, "Ctx word %d term-entity sim-based gradient: [ ", a);
                #endif
                for (c = 0; c < embedding_size; c++) {
                    term_pos_ctx_gradients[term_pos_ctx_gradient_start_ix + (a * embedding_size) + c] += (
                        local_term_entity_likelihood *
                        outer_gradient *
                        avg_weight *
                        ctx_transform_weights[c] *
                        entity_embeddings[entity_offset + c]
                    );
                    #ifdef PRINTGRADIENTS
                    fprintf(stderr, "%f ",(
                        local_term_entity_likelihood *
                        outer_gradient *
                        avg_weight *
                        ctx_transform_weights[c] *
                        entity_embeddings[entity_offset + c]
                    ));
                    #endif
                }
                #ifdef PRINTGRADIENTS
                fprintf(stderr, "]\n");
                fflush(stderr);
                #endif
            }
        }
    }

    // add in weighted term transform gradients
    #ifdef PRINTGRADIENTS
    fprintf(stderr, "Term transform gradient: [ ");
    #endif
    for (c = 0; c < embedding_size; c++) {
        term_transform_gradients[c] += (
            local_term_entity_likelihood *
            outer_gradient *
            term_embeddings[term_offset + c] *
            entity_embeddings[entity_offset + c]
        );
        #ifdef PRINTGRADIENTS
        fprintf(stderr, "%f ",(
            local_term_entity_likelihood *
            outer_gradient *
            term_embeddings[term_offset + c] *
            entity_embeddings[entity_offset + c]
        ));
        #endif
    }
    #ifdef PRINTGRADIENTS
    fprintf(stderr, "]\n");
    fflush(stderr);
    #endif

    // add in weighted ctx transform gradients
    #ifdef PRINTGRADIENTS
    fprintf(stderr, "Term transform gradient: [ ");
    #endif
    for (c = 0; c < embedding_size; c++) {
        ctx_transform_gradients[c] += (
            local_term_entity_likelihood *
            outer_gradient *
            averaged_ctx_embeddings[avg_ctx_offset + c] *
            entity_embeddings[entity_offset + c]
        );
        #ifdef PRINTGRADIENTS
        fprintf(stderr, "%f ",(
            local_term_entity_likelihood *
            outer_gradient *
            term_embeddings[term_offset + c] *
            entity_embeddings[entity_offset + c]
        ));
        #endif
    }
    #ifdef PRINTGRADIENTS
    fprintf(stderr, "]\n");
    fflush(stderr);
    #endif
}

/**
 * Add gradient from L2 regularization term
 *
 * L2 reg term is -\lambda||\theta||^2
 * Gradient d/dw_i = -2\lambda*w_i
 */
void AddRegularizationGradient(real lambda, real *embeddings, long long offset,
        real *norms, int ix, real *gradients, long gradient_start_ix,
        long long embedding_size) {
    long long c;

    // apply dimension-wise penalty
    for (c = 0; c < embedding_size; c++) {
        gradients[gradient_start_ix + c] -=
            lambda * 2 * embeddings[offset + c];
        #ifdef PRINTGRADIENTS
        fprintf(stderr, "%f ",(
            lambda * -2 * embeddings[offset + c]
        ));
        #endif
    }
}


/**
 * Apply the gradients from a single training window to all
 * model components
 *  1. Word embeddings
 *  2. Term embeddings
 *  3. Entity embeddings
 *  4. Context embeddings
 *  5. Term compositionality scores
 *  6. Term-entity likelihoods
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
        real *term_transform_weights, real *term_transform_gradients,
        real *ctx_transform_weights, real *ctx_transform_gradients,
        real *global_term_compositionality_scores, real *local_term_compositionality_scores,
        real *global_term_entity_likelihoods, real *local_term_entity_likelihoods,
        struct vocabulary *wv, int full_window_size, int target,
        int sub_window_skip, int negative, int max_num_entities, real alpha,
        long long embedding_size, bool word_burn, bool burning_in, struct model_flags *flags) {

    long long c;
    int ctx_ix, member_word_ix;
    long long ctx_offset, ctx_reg_gradient_offset, member_word_offset,
        member_word_gradient_offset;
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
        // apply gradients to the member words of the completed terms
        if (!burning_in && !flags->disable_compositionality) {
            member_word_gradient_offset = 0;
            for (i = 0; i < num_completed_terms; i++) {
                if (term_ixes[i] >= 0) {
                    for (j = 0; j < completed_term_buffer[i]->num_tokens; j++) {
                        member_word_ix = completed_term_buffer[i]->member_words[j];
                        if (member_word_ix >= 0) {
                            member_word_offset = member_word_ix * embedding_size;
                            for (c = 0; c < embedding_size; c++)
                                word_embeddings[member_word_offset + c] += (
                                //ctx_embeddings[member_word_offset + c] += (
                                    member_word_gradients[member_word_gradient_offset + c] *
                                    alpha
                                );
                            word_norms[member_word_ix] = Norm(word_embeddings,
                                member_word_offset, embedding_size);
                            //ctx_norms[member_word_ix] = Norm(ctx_embeddings,
                            //    member_word_offset, embedding_size);
                            /*
                            if (word_norms[member_word_ix] >= NORM_LIMIT) {
                                error("   WORD NORM BROKE FIRST: %f\n", word_norms[member_word_ix]);
                                exit(1);
                            }
                            */
                        }
                        member_word_gradient_offset += embedding_size;
                    }
                }
            }
        }

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
    // apply context embedding gradients from regularization
    if (!flags->disable_regularization) {
        for (i = 0; i < num_ctx; i++) {
            if (all_ctx_ixes[i] >= 0) {
                ctx_offset = all_ctx_ixes[i] * embedding_size;
                ctx_reg_gradient_offset = i * embedding_size;
                for (c = 0; c < embedding_size; c++) {
                    ctx_embeddings[ctx_offset + c]
                        += (ctx_reg_gradients[ctx_reg_gradient_offset + c] * alpha);
                }
            }
        }
    }
    // finally, update applicable norms for context embeddings
    for (i = 0; i < num_ctx; i++) {
        if (all_ctx_ixes[i] >= 0) {
            ctx_norms[all_ctx_ixes[i]] = 
                Norm(ctx_embeddings, all_ctx_ixes[i] * embedding_size, embedding_size);
            /*
            if (ctx_norms[all_ctx_ixes[i]] >= NORM_LIMIT) {
                error("   CTX NORM BROKE FIRST: %f\n", ctx_norms[all_ctx_ixes[i]]);
                exit(1);
            }
            */
        }
    }

    if (!word_burn) {
        // apply term and ctx transform gradients
        if (!burning_in && !flags->disable_term_similarity) {
            for (c = 0; c < embedding_size; c++) {
                term_transform_weights[c]
                    += (term_transform_gradients[c] * alpha);
                ctx_transform_weights[c]
                    += (ctx_transform_gradients[c] * alpha);
            }
        }

        // apply updates to the global term compositionality scores
        // update via interpolation:
        //  global_posterior(x) <- (1-alpha)global_prior(x) + (alpha)local_posterior(x)
        /*
        if (!burning_in && !flags->disable_compositionality) {
            for (i = 0; i < num_completed_terms; i++) {
                if (term_ixes[i] >= 0) {
                    global_term_compositionality_scores[term_ixes[i]] = (
                        (
                            (1 - alpha) *
                            global_term_compositionality_scores[term_ixes[i]]
                        ) +
                        (
                            alpha *
                            local_term_compositionality_scores[i]
                        )
                    );
                }
            }
        }
        */

        // apply updates to context-independent (global) term->entity likelihoods
        // update via interpolation:
        //  global_posterior(x) <- (1-alpha)global_prior(x) + (alpha)local_posterior(x)
        if (!burning_in && !flags->disable_likelihoods) {
            for (i = 0; i < num_completed_terms; i++) {
                if (term_ixes[i] >= 0) {
                    term_entity_block_start = i * max_num_entities;
                    for (j = 0; j < entities_per_term[i]; j++) {
                        if (entity_ixes[term_entity_block_start + j] >= 0) {
                            global_term_entity_likelihoods[(term_ixes[i] * max_num_entities) + j] = (
                                (
                                    (1 - alpha) *
                                    global_term_entity_likelihoods[(term_ixes[i] * max_num_entities) + j]
                                ) + 
                                (
                                    alpha *
                                    local_term_entity_likelihoods[term_entity_block_start + j]
                                )
                            );
                        }
                    }
                }
            }
        }
    }
}


/**
 * Update model parameters for one training step.  Consists of three phases:
 *   1. Calculate context-sensitive posteriors for:
 *      a) term latency
 *      b) entity likelihood
 *   2. Calculate gradients, based on:
 *      a) positive/negative contexts
 *      b) term compositionality
 *      c) entity-term similarity
 *      d) L2-regularization
 *   3. Apply batched gradients
 */
void LearningStep(int *masked_word_context_window, int target, int full_window_size,
        int sub_window_skip, struct term_annotation **completed_term_buffer, int num_completed_terms,
        int *sampled_completed_term_ixes, int *word_negative_samples, int *term_negative_samples,
        struct vocabulary *wv, struct vocabulary *tv, struct vocabulary *ev, struct entity_map *termmap,
        struct term_monogamy_map *monomap,
        int max_num_entities, real *word_embeddings, real *term_embeddings, real *entity_embeddings,
        real *ctx_embeddings, real *word_norms, real *term_norms, real *entity_norms, real *ctx_norms,
        int *entity_update_counters, int *ctx_update_counters, real *global_term_compositionality_scores,
        real *global_term_entity_likelihoods, real *term_transform_weights, real *ctx_transform_weights,
        real alpha, long long embedding_size, int negative, real lambda, bool word_burn, bool burning_in,
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

    real *local_term_latency_scores = NULL,
        *local_term_compositionality_scores = NULL,
        *local_term_entity_likelihoods = NULL;
    if (!flags->disable_terms || !flags->disable_entities) {
        local_term_latency_scores = MallocOrDie(num_completed_terms * sizeof(real), "local term latency scores");
        local_term_compositionality_scores = MallocOrDie(num_completed_terms * sizeof(real), "local term compositionality scores");
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
    long long member_word_gradient_offset;

    real *term_transform_gradients = NULL, *ctx_transform_gradients = NULL;
    if (!flags->disable_term_similarity) {
        term_transform_gradients = MallocOrDie(embedding_size * sizeof(real), "term transform gradients");
        ctx_transform_gradients = MallocOrDie(embedding_size * sizeof(real), "ctx transform gradients");
    }



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
        //CalculateAverageWordEmbeddings(completed_term_buffer,
        //    num_completed_terms, sampled_completed_term_ixes,
        //    word_embeddings, embedding_size, averaged_word_embeddings);
        if (!burning_in && !flags->disable_compositionality && !flags->disable_terms) {
            CombineWeightedMemberWordEmbeddings(completed_term_buffer,
                num_completed_terms, sampled_completed_term_ixes,
                word_embeddings, monomap, embedding_size, combined_word_embeddings);
        }

        if (!burning_in && (!flags->disable_latency || !flags->disable_term_similarity)) {
            CalculateAverageContextEmbeddings(completed_term_buffer,
                num_completed_terms, sampled_completed_term_ixes,
                ctx_embeddings, embedding_size, sub_window_skip,
                full_window_size - sub_window_skip, target,
                averaged_ctx_embeddings);
        }


        // CALCULATE ALL THE LOCAL SCORES!1!
        if (!burning_in && !flags->disable_likelihoods) {
            CalculateLocalTermEntityLikelihoods(completed_term_buffer,
                num_completed_terms, sampled_completed_term_ixes, max_num_entities,
                entities_per_term, global_term_entity_likelihoods, entity_embeddings, entity_norms,
                entity_ixes, entity_offsets, entity_pos_ctx_dots, ctx_norms, full_window_size,
                target, embedding_size, local_term_entity_likelihoods);
        } else {
            for (i = 0; i < num_completed_terms; i++) {
                term_entity_block_start = i * max_num_entities;
                for (j = 0; j < entities_per_term[i]; j++)
                    local_term_entity_likelihoods[term_entity_block_start + j] = 1/(real)entities_per_term[i];
                for (j = entities_per_term[i]; j < max_num_entities; j++)
                    local_term_entity_likelihoods[term_entity_block_start + j] = 0;
            }
        }

        if (!burning_in && !flags->disable_latency) {
            CalculateLocalTermLatencyScores(completed_term_buffer, num_completed_terms,
                sampled_completed_term_ixes, max_num_entities, entities_per_term,
                entity_embeddings, entity_norms, entity_ixes, averaged_ctx_embeddings,
                embedding_size, local_term_latency_scores);
        } else {
            for (i = 0; i < num_completed_terms; i++)
                local_term_latency_scores[i] = 1;
        }

        /*
        if (!burning_in && !flags->disable_compositionality) {
            CalculateLocalTermCompositionalityScores(completed_term_buffer,
                num_completed_terms, sampled_completed_term_ixes, embedding_size,
                term_embeddings, term_norms, averaged_word_embeddings,
                local_term_compositionality_scores);
        } else {
            for (i = 0; i < num_completed_terms; i++)
                local_term_compositionality_scores[i] = 0;
        }
        */
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
    if (!flags->disable_regularization) {
        for (c = 0; c < num_ctx * embedding_size; c++)
            ctx_reg_gradients[c] = 0;
    }
    if (!flags->disable_term_similarity) {
        for (c = 0; c < embedding_size; c++) {
            term_transform_gradients[c] = 0;
            ctx_transform_gradients[c] = 0;
        }
    }

    // (1) Word gradients
    if (!flags->disable_words && word_ix >= 0) {
        
        // (1.1) mini-batched word gradients from contexts
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

        // (1.2) word regularization gradient
        if (!flags->disable_regularization) {
            #ifdef PRINTGRADIENTS
            fprintf(stderr, "Word regularization gradient: [ ");
            #endif
            AddRegularizationGradient(lambda, word_embeddings, word_offset,
                word_norms, word_ix, word_gradient, 0, embedding_size);
            #ifdef PRINTGRADIENTS
            fprintf(stderr, "]\n");
            fflush(stderr);
            #endif
        }
    }

    if (!word_burn && (!flags->disable_terms || !flags->disable_entities)) {
        // (2-3) Term/entity gradients
        member_word_gradient_offset = 0;
        for (i = 0; i < num_completed_terms; i++) {
            if (term_ixes[i] >= 0) {
                term_block_start = i * full_window_size;
                term_ns_block_start = i * full_window_size * negative;
                term_gradient_offset = i * embedding_size;
                
                // (2.1) mini-batched term embedding gradients from contexts
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
                        term_ns_block_start * embedding_size, embedding_size,
                        local_term_latency_scores[i]);
                    #ifdef PRINTGRADIENTS
                    fprintf(stderr, "]\n");
                    fflush(stderr);
                    #endif
                }

                // (2.2) add in term->word compositionality gradients
                // [does not use term latency score]
                if (!burning_in && !flags->disable_compositionality) {
                    AddMemberWordBasedGradients(term_embeddings, term_ixes[i],
                        term_offsets[i], combined_word_embeddings,
                        i*embedding_size, completed_term_buffer[i]->member_words,
                        completed_term_buffer[i]->num_tokens, monomap,
                        term_gradients, term_gradient_offset,
                        member_word_gradients, member_word_gradient_offset,
                        embedding_size);

                    member_word_gradient_offset += 
                        (completed_term_buffer[i]->num_tokens * embedding_size);
                }

                // (2.3) term regularization gradient
                if (!flags->disable_terms || !flags->disable_regularization) {
                    #ifdef PRINTGRADIENTS
                    fprintf(stderr, "Term regularization gradient: [ ");
                    #endif
                    AddRegularizationGradient(lambda, term_embeddings, term_offsets[i],
                        term_norms, term_ixes[i], term_gradients, term_gradient_offset,
                        embedding_size);
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

                            // (3.1) mini-batched entity gradients from contexts
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
                                    local_term_latency_scores[i] * local_term_entity_likelihoods[term_entity_block_start + j]);
                                #ifdef PRINTGRADIENTS
                                fprintf(stderr, "]\n");
                                fflush(stderr);
                                #endif
                            }

                            // (3.2) entity-term similarity gradients
                            if (!burning_in && !flags->disable_term_similarity) {
                                AddTermEntitySimilarityBasedGradients(term_embeddings, term_offsets[i],
                                    entity_embeddings, entity_offsets[term_entity_block_start + j],
                                    averaged_ctx_embeddings, i*embedding_size, completed_term_buffer[i]->contexts,
                                    sub_window_skip, full_window_size - sub_window_skip, target, term_transform_weights,
                                    ctx_transform_weights, local_term_entity_likelihoods[term_entity_block_start + j],
                                    entity_gradients, entity_gradient_offset, term_gradients, term_gradient_offset,
                                    term_pos_ctx_gradients, i * full_window_size * embedding_size,
                                    term_transform_gradients, ctx_transform_gradients, embedding_size);
                            }

                            // (3.3) entity regularization gradient
                            if (!flags->disable_regularization) {
                                // First, check to see if we've already grabbed a regularization
                                // gradient for this entity
                                if (entity_update_counters[entity_ixes[term_entity_block_start + j]] == 0) {
                                    #ifdef PRINTGRADIENTS
                                    fprintf(stderr, "Entity regularization gradient: [ ");
                                    #endif
                                    AddRegularizationGradient(lambda, entity_embeddings,
                                        entity_offsets[term_entity_block_start + j], entity_norms,
                                        entity_ixes[term_entity_block_start + j], entity_gradients,
                                        entity_gradient_offset, embedding_size);
                                    #ifdef PRINTGRADIENTS
                                    fprintf(stderr, "]\n");
                                    fflush(stderr);
                                    #endif

                                    // flag that it's been regularized
                                    entity_update_counters[entity_ixes[term_entity_block_start + j]] = 1;
                                } else {
                                    // if we've already grabbed it, add zero to the gradient
                                    for (c = 0; c < embedding_size; c++)
                                        entity_gradients[entity_gradient_offset + c] += 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // (4) Additional context regularization gradients
    if (!flags->disable_regularization) {
        for (i = 0; i < num_ctx; i++) {
            if (all_ctx_ixes[i] >= 0) {
                // (4.1) context regularization gradient
                // check to see if we've already grabbed a regularization gradient 
                // for this context word
                if (ctx_update_counters[all_ctx_ixes[i]] == 0) {
                    #ifdef PRINTGRADIENTS
                    fprintf(stderr, "Ctx regularization gradient: [ ");
                    #endif
                    AddRegularizationGradient(lambda, ctx_embeddings,
                        all_ctx_ixes[i] * embedding_size, ctx_norms,
                        all_ctx_ixes[i], ctx_reg_gradients, i*embedding_size,
                        embedding_size);
                    #ifdef PRINTGRADIENTS
                    fprintf(stderr, "]\n");
                    fflush(stderr);
                    #endif

                    // flag that it's been regularized
                    ctx_update_counters[all_ctx_ixes[i]] = 1;
                } else {
                    // if we've already grabbed it, just add zero to the gradient
                    for (c = 0; c < embedding_size; c++)
                        ctx_reg_gradients[(i*embedding_size) + c] += 0;
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
        all_ctx_ixes, num_ctx, term_transform_weights, term_transform_gradients,
        ctx_transform_weights, ctx_transform_gradients, global_term_compositionality_scores,
        local_term_compositionality_scores, global_term_entity_likelihoods,
        local_term_entity_likelihoods, wv, full_window_size, target, sub_window_skip,
        negative, max_num_entities, alpha, embedding_size, word_burn, burning_in, flags);


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
    FreeAndNull((void *)&local_term_latency_scores);
    FreeAndNull((void *)&local_term_compositionality_scores);
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

    FreeAndNull((void *)&term_transform_gradients);
    FreeAndNull((void *)&ctx_transform_gradients);
}
