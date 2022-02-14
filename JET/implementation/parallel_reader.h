#ifndef _parallel_reader_h
#define _parallel_reader_h

void AllocateBuffers(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer, int buffer_size,
        int max_string_len);
void DestroyBuffers(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer, int buffer_size);

void PreloadBuffers_ByteTracking(struct indexed_string **word_buffer,
        long long *word_byte_buffer, struct term_annotation **passive_term_buffer,
        long long *passive_term_byte_buffer, int buffer_size, FILE *word_hook,
        FILE *term_hook, int max_string_len);
void PreloadBuffers_VocabLookup(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer, int buffer_size,
        FILE *word_hook, FILE *term_hook, int max_string_len,
        struct vocabulary *word_vocab, struct vocabulary *term_vocab);
void PreloadBuffers(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer, int buffer_size,
        FILE *word_hook, FILE *term_hook, int max_string_len);

void ProcessTermBuffers_VocabLookup(long long *tokens_since_last_annotation,
        struct term_annotation **passive_term_buffer, int *passive_term_buffer_ix,
        int buffer_size,
        struct term_annotation **active_term_buffer, int *num_active_terms,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        FILE *annothook, struct vocabulary *term_vocab);

int ParallelReadStep_ByteTracking(struct indexed_string **word_buffer,
        long long *word_byte_buffer, struct term_annotation **passive_term_buffer,
        long long *passive_term_byte_buffer, struct term_annotation **active_term_buffer,
        int buffer_size, int *word_buffer_ix, int words_to_keep,
        int *passive_term_buffer_ix, int *num_active_terms,
        long long *word_buffer_start_ix, long long *tokens_since_last_annotation,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        int max_string_len, FILE *plainhook, FILE *annothook);
int ParallelReadStep_VocabLookup(struct indexed_string **word_buffer, 
        struct term_annotation **passive_term_buffer,
        struct term_annotation **active_term_buffer, int buffer_size,
        int *word_buffer_ix, int words_to_keep,
        int *passive_term_buffer_ix, int *num_active_terms,
        long long *word_buffer_start_ix, long long *tokens_since_last_annotation,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        int max_string_len, FILE *plainhook, FILE *annothook,
        struct vocabulary *word_vocab, struct vocabulary *term_vocab);
int ParallelReadStep(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer,
        struct term_annotation **active_term_buffer, int buffer_size,
        int *word_buffer_ix, int words_to_keep, 
        int *passive_term_buffer_ix, int *num_active_terms,
        long long *word_buffer_start_ix, long long *tokens_since_last_annotation,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        int max_string_len, FILE *plainhook, FILE *annothook);

#endif
