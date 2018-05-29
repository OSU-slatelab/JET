#ifndef _model_h
#define _model_h

typedef float real;                    // Precision of float numbers

struct model_flags {
    bool disable_words;
    bool disable_terms;
    bool disable_entities;
};

struct hyperparameters {
    char *plaintext_corpus_file;
    char *corpus_annotations_file;
    int numiters;
    int window;
    int min_count;
    long long embedding_size;
    real alpha;
    long long alpha_schedule_interval;
    real downsampling_rate;
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
};

bool RollToDownsample(real *downsampling_table, int ix);

void InitModelFlags(struct model_flags **flags);
void DestroyModelFlags(struct model_flags **flags);

void InitializeModel(real **word_embeddings, real **term_embeddings, real **entity_embeddings,
        real **ctx_embeddings, real **word_norms, real **term_norms, real **entity_norms,
        real **ctx_norms,
        struct vocabulary *wv, struct vocabulary *tv, struct vocabulary *ev, struct entity_map *em,
        long long embedding_size, int **unitable, real **word_downsampling_table,
        real **term_downsampling_table, real downsampling_rate);
void DestroyModel(real **word_embeddings, real **term_embeddings, real **entity_embeddings,
        real **ctx_embeddings, real **word_norms, real **term_norms, real **entity_norms,
        real **ctx_norms,
        int **unitable, real **word_downsampling_table, real **term_downsampling_table);

void LearningStep(int *masked_word_context_window, int target, int full_window_size,
        int sub_window_skip, struct term_annotation **completed_term_buffer, int num_completed_terms,
        int *sampled_completed_term_ixes, int *word_negative_samples, int *term_negative_samples,
        struct vocabulary *wv, struct vocabulary *tv, struct vocabulary *ev, struct entity_map *termmap,
        int max_num_entities, real *word_embeddings, real *term_embeddings, real *entity_embeddings,
        real *ctx_embeddings, real *word_norms, real *term_norms, real *entity_norms, real *ctx_norms,
        int *entity_update_counters, int *ctx_update_counters,
        real alpha, long long embedding_size, int negative, struct model_flags *flags);

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
void CalculateAverageContextEmbeddings(struct term_annotation **completed_term_buffer,
        int num_completed_terms, int *sampled_completed_term_ixes, real *ctx_embeddings,
        long long embedding_size, int window_start, int window_end, int target,
        real *averaged_ctx_embeddings);
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
int RandomSubwindowSkip(int window_size);
void InitDownsamplingTable(real **downsampling_table, struct vocabulary *v, real downsampling_rate);
void DestroyDownsamplingTable(real **downsampling_table);
/** End component methods **/

#endif
