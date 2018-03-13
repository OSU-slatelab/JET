#ifndef _model_h
#define _model_h

typedef float real;                    // Precision of float numbers

struct model_flags {
    bool disable_compositionality;
    bool disable_likelihoods;
    bool disable_term_similarity;
    bool disable_latency;
    bool disable_regularization;
    bool disable_words;
    bool disable_terms;
    bool disable_entities;
};

struct hyperparameters {
    char *plaintext_corpus_file;
    char *corpus_annotations_file;
    int numiters;
    int burn_in_iters;
    int word_burn_iters;
    int window;
    int min_count;
    long long embedding_size;
    real alpha;
    long long alpha_schedule_interval;
    real downsampling_rate;
    real lambda;
    long random_seed;
    struct model_flags *flags;
    int num_threads;

    // misc files
    char *map_file;
    char *str_map_sep;
    char *term_strmap_file;
    char *thread_config_file;

    // vocab
    char *wvocab_file;
    char *tvocab_file;

    // model
    char *word_vectors_file;
    char *term_vectors_file;
    char *entity_vectors_file;
    char *context_vectors_file;
    char *term_compositionality_file;
    char *term_entity_likelihood_file;
    char *interpolation_weights_file;
};

bool RollToDownsample(real *downsampling_table, int ix);

void InitModelFlags(struct model_flags **flags);
void DestroyModelFlags(struct model_flags **flags);

void InitializeModel(real **word_embeddings, real **term_embeddings, real **entity_embeddings,
        real **ctx_embeddings, real **word_norms, real **term_norms, real **entity_norms,
        real **ctx_norms, real **term_transform_weights, real **ctx_transform_weights,
        real **global_term_compositionality_scores, real **global_term_entity_likelihoods,
        struct vocabulary *wv, struct vocabulary *tv, struct vocabulary *ev, struct entity_map *em,
        long long embedding_size, int **unitable, real **word_downsampling_table,
        real **term_downsampling_table, real downsampling_rate);
void DestroyModel(real **word_embeddings, real **term_embeddings, real **entity_embeddings,
        real **ctx_embeddings, real **word_norms, real **term_norms, real **entity_norms,
        real **ctx_norms, real **term_transform_weights, real **ctx_transform_weights,
        real **global_term_compositionality_scores, real **global_term_entity_likelihoods,
        int **unitable, real **word_downsampling_table, real **term_downsampling_table);

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
        struct model_flags *flags);

/** Component methods (accessible for testing) **/
real DotProduct(real *embeds_a, long long offset_a, real *embeds_b, long long offset_b,
        long long embedding_size);
real Norm(real *embeds_a, long long offset_a, long long embedding_size);
real CosineSimilarity(real *embeds_a, long long offset_a, real *embeds_b,
        long long offset_b, real norm_a, real norm_b, long long embedding_size);
real CalculateLogSigmoidOuterGradient(real dot_product, int label);
void CalculateContextDotProducts(real *embeddings, long long trg_offset,
        real *ctx_embeddings, int *context_window, int *negative_samples,
        int negative, long ns_start_ix, int full_window_size, int target,
        int window_start, int window_end, long long embedding_size,
        real *pos_ctx_dots, long long pos_ctx_start_ix,
        real *neg_ctx_dots, long long neg_ctx_start_ix);
void CombineWeightedMemberWordEmbeddings(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, real *word_embeddings,
        struct term_monogamy_map *monomap, long long embedding_size,
        real *combined_word_embeddings);
void CalculateAverageContextEmbeddings(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, real *ctx_embeddings,
        long long embedding_size, int window_start, int window_end, int target,
        real *averaged_ctx_embeddings);
void CalculateLocalTermLatencyScores(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, int max_num_entities,
        int *entities_per_term, real *entity_embeddings, real *entity_norms,
        int *entity_ixes, real *averaged_ctx_embeddings, long long embedding_size,
        real *local_term_latency_scores);
/*
void CalculateLocalTermCompositionalityScores(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, long long embedding_size,
        real *term_embeddings, real *term_norms, real *word_embeddings,
        real *local_term_compositionality_scores);
*/
void CalculateLocalTermEntityLikelihoods(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, int max_num_entities,
        int *entities_per_term, real *global_term_entity_likelihoods, real *entity_embeddings,
        real *entity_norms, int *entity_ixes, long long *entity_offsets,
        real *entity_pos_ctx_dots, real *ctx_norms, int full_window_size, int target,
        long long embedding_size, real *local_term_entity_likelihoods);
void GetNegativeSamples(int negative, int *negative_samples, int ns_start_ix, int *pos_ctx_ixes,
        int window_start, int window_end, int target, int *unitable);
void AddContextBasedGradients(real *embeddings, long long trg_offset,
        real *ctx_embeddings, int *context_window,
        int *negative_samples, int negative, long ns_start_ix,
        int full_window_size, int target, int window_start, int window_end,
        real *pos_ctx_dots, long long pos_ctx_start_ix,
        real *neg_ctx_dots, long long neg_ctx_start_ix,
        real *gradients, long gradient_start_ix, real *pos_ctx_gradients,
        long pos_ctx_gradient_start_ix, real *neg_ctx_gradients,
        long neg_ctx_gradient_start_ix, long long embedding_size, real constant_weight);
void AddMemberWordBasedGradients(real *term_embeddings, int term_ix,
        long long term_offset, real *combined_word_embeddings,
        long long word_comb_offset, int *member_words, int num_tokens,
        struct term_monogamy_map *monomap, real *term_gradients,
        long term_gradient_start_ix, real *member_word_gradients,
        long member_word_gradient_start_ix, long long embedding_size);
void AddTermEntitySimilarityBasedGradients(real *term_embeddings, long long term_offset,
        real *entity_embeddings, long long entity_offset, real *averaged_ctx_embeddings,
        long long avg_ctx_offset, int *term_pos_ctx_ixes, int window_start, int window_end,
        int target, real *term_transform_weights, real *ctx_transform_weights,
        real local_term_entity_likelihood, real *entity_gradients, long entity_gradient_start_ix,
        real *term_gradients, long term_gradient_start_ix, real *term_pos_ctx_gradients,
        long long term_pos_ctx_gradient_start_ix, real *term_transform_gradients,
        real *ctx_transform_gradients, long long embedding_size);
void AddRegularizationGradient(real lambda, real *embeddings, long long offset,
        real *norms, int ix, real *gradients, long gradient_start_ix,
        long long embedding_size);
int RandomSubwindowSkip(int window_size);
void InitDownsamplingTable(real **downsampling_table, struct vocabulary *v, real downsampling_rate);
void DestroyDownsamplingTable(real **downsampling_table);
/** End component methods **/

#endif
