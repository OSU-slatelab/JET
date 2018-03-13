#ifndef _context_manager_h
#define _context_manager_h

void DownsampleWordAndTerms(struct indexed_string **word_buffer, int next_word_ix,
        struct term_annotation **completed_term_buffer, int num_completed_terms,
        real *word_downsampling_table, real *term_downsampling_table,
        int *sampled_next_word_ix, int *sampled_completed_term_ixes);

void UpdateContextWindows(int *word_context_window, int *masked_word_context_window,
        int sampled_next_word_ix, int num_active_terms, struct term_annotation **active_term_buffer,
        int num_completed_terms, struct term_annotation **completed_term_buffer,
        int *sampled_completed_term_ixes, bool *finishing_sentence, int full_window_size,
        int target);

void InitializeBuffersAndContexts(struct indexed_string **word_buffer, int *word_buffer_ix,
        struct term_annotation **passive_term_buffer, int *passive_term_buffer_ix,
        struct term_annotation **active_term_buffer, int *num_active_terms,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        int *sampled_next_word_ix, int **sampled_completed_term_ixes,
        int *word_context_window, int *masked_word_context_window,
        long long *tokens_since_last_annotation, bool *finishing_sentence, int buffer_size,
        int full_window_size, int window_size, int target, real *word_downsampling_table,
        real *term_downsampling_table, FILE *plnhook, FILE *annhook,
        struct vocabulary *word_vocab, struct vocabulary *term_vocab);

#endif
