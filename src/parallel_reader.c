#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include "logging.h"
#include "io.h"
#include "vocab.h"
#include "parallel_reader.h"

/**
 * Allocate memory for parallel reading buffers
 */
void AllocateBuffers(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer, int buffer_size, int max_string_len) {
    for (int a = 0; a < buffer_size; a++) {
        // initialize the word
        word_buffer[a] = malloc(sizeof(struct indexed_string));
        if (word_buffer[a] == NULL) {
            error("ERROR >>> Failed to allocate struct memory in word buffer\n");
            exit(1);
        }
        // and its string
        word_buffer[a]->string = calloc(max_string_len, sizeof(char));
        word_buffer[a]->vocab_index = -1;
        if (word_buffer[a]->string == NULL) {
            error("ERROR >>> Failed to allocate string memory in word buffer\n");
            exit(1);
        }
        // initialize the passive term annotation
        passive_term_buffer[a] = malloc(sizeof(struct term_annotation));
        if (passive_term_buffer[a] == NULL) {
            error("ERROR >>> Failed to allocate annotation struct memory in term buffer\n");
            exit(1);
        }
        // its specified term
        passive_term_buffer[a]->term = malloc(sizeof(struct indexed_string));
        if (passive_term_buffer[a]->term == NULL) {
            error("ERROR >>> Failed to allocate term struct memory in term buffer\n");
            exit(1);
        }
        // and its string
        passive_term_buffer[a]->term->string = calloc(max_string_len, sizeof(char));
        passive_term_buffer[a]->term->vocab_index = -1;
        if (passive_term_buffer[a]->term->string == NULL) {
            error("ERROR >>> Failed to allocate string memory in term buffer\n");
            exit(1);
        }
    }
}

/**
 * Free memory from parallel reading buffers
 */
void DestroyBuffers(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer, int buffer_size) {
    for (int a = 0; a < buffer_size; a++) {
        // free from word buffer
        if (word_buffer[a] != NULL && word_buffer[a]->string != NULL) free(word_buffer[a]->string);
        if (word_buffer[a] != NULL) free(word_buffer[a]);

        // free from term buffer
        if (passive_term_buffer[a] != NULL) {
            // term (string and index)
            if (passive_term_buffer[a]->term != NULL) {
                if (passive_term_buffer[a]->term->string != NULL) {
                    free(passive_term_buffer[a]->term->string);
                    passive_term_buffer[a]->term->string = NULL;
                }
                free(passive_term_buffer[a]->term);
                passive_term_buffer[a]->term = NULL;
            }
            // contexts
            if (passive_term_buffer[a]->contexts != NULL) {
                free(passive_term_buffer[a]->contexts);
                passive_term_buffer[a]->contexts = NULL;
            }
            // member words
            if (passive_term_buffer[a]->member_words != NULL) {
                free(passive_term_buffer[a]->member_words);
                passive_term_buffer[a]->member_words = NULL;
            }

            free(passive_term_buffer[a]);
            passive_term_buffer[a] = NULL;
        }
    }
}

/**
 * Clear and refill the plaintext word buffer.
 * Reuses memory allocated for each word.
 */
//@{
void ShiftWordBuffer_Master(struct indexed_string **word_buffer, long long *word_byte_buffer,
        int buffer_size, int words_to_keep, FILE *fin, int max_string_len,
        struct vocabulary *word_vocab) {
    int index, i;
    struct indexed_string *tmp_word;
    long long tmp_byte;

    // copy the last words_to_keep words down to the head of the buffer
    for (i = 0; i < words_to_keep; i++) {
        index = buffer_size - words_to_keep + i; 
        // save the pointer currently in the ith slot
        tmp_word = word_buffer[i];
        if (word_byte_buffer != NULL) tmp_byte = word_byte_buffer[i];
        // copy the pointer in index into the ith slot
        word_buffer[i] = word_buffer[index];
        if (word_byte_buffer != NULL) word_byte_buffer[i] = word_byte_buffer[index];
        // and move the temporary pointer into the index slot (swapped)
        word_buffer[index] = tmp_word;
        if (word_byte_buffer != NULL) word_byte_buffer[index] = tmp_byte;
    }
    // now read in new words from the file
    for (i = words_to_keep; i < buffer_size; i++) {
        if (word_byte_buffer != NULL) word_byte_buffer[i] = ftell(fin);
        if (!feof(fin)) {
            ReadCorpusWord(word_buffer[i]->string, fin, max_string_len);
        } else {
            word_buffer[i]->string[0] = 0;
        }
    }
    // then, if specified, find the vocabulary index for each new word
    if (word_vocab != NULL) {
        for (i = words_to_keep; i < buffer_size; i++) {
            if (word_buffer[i]->string[0] == 0) {
                word_buffer[i]->vocab_index = -1;
            } else {
                word_buffer[i]->vocab_index = SearchVocab(word_vocab, word_buffer[i]->string);
            }
        }
    }
}
void ShiftWordBuffer_VocabLookup(struct indexed_string **word_buffer, int buffer_size,
        int words_to_keep, FILE *fin, int max_string_len, struct vocabulary *word_vocab) {
    ShiftWordBuffer_Master(word_buffer, NULL, buffer_size, words_to_keep, fin,
        max_string_len, word_vocab);
}
void ShiftWordBuffer_ByteTracking(struct indexed_string **word_buffer,
        long long *word_byte_buffer, int buffer_size, int words_to_keep, FILE *fin,
        int max_string_len) {
    ShiftWordBuffer_Master(word_buffer, word_byte_buffer, buffer_size, words_to_keep, 
        fin, max_string_len, NULL);
}
void ShiftWordBuffer(struct indexed_string **word_buffer, int buffer_size,
        int words_to_keep, FILE *fin, int max_string_len) {
    ShiftWordBuffer_Master(word_buffer, NULL, buffer_size, words_to_keep, fin,
        max_string_len, NULL);
}
//@}

/**
 * Clear all completed terms from the passive term buffer, and
 * read in upcoming terms.
 * Moves all active terms to the start of the buffer.
 * Reuses memory allocated for each term.
 */
//@{
void ShiftPassiveTermBuffer_Master(struct term_annotation **passive_term_buffer,
        long long *passive_term_byte_buffer, int buffer_size,
        struct term_annotation **active_term_buffer, int num_active_terms,
        FILE *fin, struct vocabulary *term_vocab) {
    int index, i;
    struct term_annotation *tmp_annot;
    long long tmp_byte;

    // copy the current active terms down to the head of the buffer
    // (to keep them in active memory)
    for (i = 0; i < num_active_terms; i++) {
        // get the active term's current position in the passive term buffer
        index = active_term_buffer[i]->buffer_index;
        // save the pointer currently in the ith slot
        tmp_annot = passive_term_buffer[i];
        if (passive_term_byte_buffer != NULL) tmp_byte = passive_term_byte_buffer[i];
        // copy the pointer in index into the ith slot
        passive_term_buffer[i] = passive_term_buffer[index];
        if (passive_term_byte_buffer != NULL) passive_term_byte_buffer[i] = passive_term_byte_buffer[index];
        // and move the temporary pointer into the index slot (swapped)
        passive_term_buffer[index] = tmp_annot;
        if (passive_term_byte_buffer != NULL) passive_term_byte_buffer[index] = tmp_byte;
        // finally, update both of their buffer index positions
        passive_term_buffer[i]->buffer_index = i;            // new home for the active term
        passive_term_buffer[index]->buffer_index = index;    // old home for the active term
    }
    // read in new annotations to fill the rest of the buffer
    for (i = num_active_terms; i < buffer_size; i++) {
        if (passive_term_byte_buffer != NULL) passive_term_byte_buffer[i] = ftell(fin);
        if (!feof(fin)) {
            ReadAnnotation(passive_term_buffer[i], fin);
        } else {
            passive_term_buffer[i]->start_offset = -1; // unreachable
            passive_term_buffer[i]->term->string[0] = 0;
        }
        passive_term_buffer[i]->buffer_index = i;
    }
    // now, for all new terms in the buffer, look up their vocabulary indices
    if (term_vocab != NULL) {
        for (i = num_active_terms; i < buffer_size; i++) {
            if (passive_term_buffer[i]->term->string[0] == 0) {
                passive_term_buffer[i]->term->vocab_index = -1;
            } else {
                passive_term_buffer[i]->term->vocab_index =
                    SearchVocab(term_vocab, passive_term_buffer[i]->term->string);
            }
        }
    }
}
void ShiftPassiveTermBuffer_VocabLookup(struct term_annotation **passive_term_buffer,
        int buffer_size, struct term_annotation **active_term_buffer, 
        int num_active_terms, FILE *fin, struct vocabulary *term_vocab) {
     ShiftPassiveTermBuffer_Master(passive_term_buffer, NULL, buffer_size,
        active_term_buffer, num_active_terms, fin, term_vocab);
}
void ShiftPassiveTermBuffer_ByteTracking(struct term_annotation **passive_term_buffer,
        long long *passive_term_byte_buffer, int buffer_size,
        struct term_annotation **active_term_buffer, int num_active_terms,
        FILE *fin) {
     ShiftPassiveTermBuffer_Master(passive_term_buffer, passive_term_byte_buffer,
        buffer_size, active_term_buffer, num_active_terms, fin, NULL);
}
void ShiftPassiveTermBuffer(struct term_annotation **passive_term_buffer,
        int buffer_size, struct term_annotation **active_term_buffer, 
        int num_active_terms, FILE *fin) {
     ShiftPassiveTermBuffer_Master(passive_term_buffer, NULL, buffer_size,
        active_term_buffer, num_active_terms, fin, NULL);
}
//@}

/**
 * Move all active terms one step forward in the buffer.
 * Don't have to worry about memory management here, because these pointers are
 * managed in FillTermBuffer.
 */
void ShiftActiveTermBuffer(struct term_annotation **term_buffer, int start_ix, int num_active_terms) {
    for (int i = start_ix+1; i < num_active_terms; i++) {
        term_buffer[i-1] = term_buffer[i];
    }
    term_buffer[num_active_terms - 1] = NULL;
}

/**
 * Load the initial contents into reading buffers
 */
//@{
void PreloadBuffers_ByteTracking(struct indexed_string **word_buffer,
        long long *word_byte_buffer, struct term_annotation **passive_term_buffer,
        long long *passive_term_byte_buffer, int buffer_size, FILE *word_hook,
        FILE *term_hook, int max_string_len) {
    ShiftWordBuffer_ByteTracking(word_buffer, word_byte_buffer, buffer_size, 0,
        word_hook, max_string_len);
    ShiftPassiveTermBuffer_ByteTracking(passive_term_buffer, passive_term_byte_buffer,
        buffer_size, NULL, 0, term_hook);
}
void PreloadBuffers_VocabLookup(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer, int buffer_size,
        FILE *word_hook, FILE *term_hook, int max_string_len,
        struct vocabulary *word_vocab, struct vocabulary *term_vocab) {
    ShiftWordBuffer_VocabLookup(word_buffer, buffer_size, 0, word_hook,
        max_string_len, word_vocab);
    ShiftPassiveTermBuffer_VocabLookup(passive_term_buffer, buffer_size,
        NULL, 0, term_hook, term_vocab);
}
void PreloadBuffers(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer, int buffer_size,
        FILE *word_hook, FILE *term_hook, int max_string_len) {
    ShiftWordBuffer(word_buffer, buffer_size, 0, word_hook, max_string_len);
    ShiftPassiveTermBuffer(passive_term_buffer, buffer_size, NULL,
        0, term_hook);
}
//@}


/**
 * After moving forward one word in the corpus, process updates to terms
 * at all stages.
 *   (1) If we're starting any new terms, add them to the active term buffer
 *   (2) For all active terms, mark that another token has been processed
 *   (3) For any active terms that have processed all their tokens, move
 *       them to the completed term buffer.
 */
//@{
void ProcessTermBuffers_Master(long long *tokens_since_last_annotation,
        struct term_annotation **passive_term_buffer, int *passive_term_buffer_ix,
        int buffer_size, long long *passive_term_byte_buffer,
        struct term_annotation **active_term_buffer, int *num_active_terms,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        FILE *annothook, struct vocabulary *term_vocab) {
    int i;

    // check if we're starting new term(s)
    while (*tokens_since_last_annotation == passive_term_buffer[*passive_term_buffer_ix]->start_offset) {
        // copy the term into the active buffer
        active_term_buffer[*num_active_terms] = passive_term_buffer[*passive_term_buffer_ix];
        *num_active_terms = *num_active_terms + 1;
        // mark that we're starting to track this term
        active_term_buffer[(*num_active_terms)-1]->tokens_so_far = 0;
        // drop the offset counter back to 0
        *tokens_since_last_annotation = 0;
        // and bump the passive buffer forward one
        *passive_term_buffer_ix = *passive_term_buffer_ix + 1;

        // refill term buffer as needed
        if (*passive_term_buffer_ix == buffer_size) {
            ShiftPassiveTermBuffer_Master(passive_term_buffer,
                passive_term_byte_buffer, buffer_size, active_term_buffer,
                *num_active_terms, annothook, term_vocab);
            *passive_term_buffer_ix = *num_active_terms;   // reset the local counter
        }
    }

    // update the token tracking for all active terms
    //if (*num_active_terms >= 3) printf("\nACTIVE TERM STATUS\n");
    for (i = 0; i < *num_active_terms; i++) {
        //if (*num_active_terms >= 3) printf("  Term %d (%p; %s): Sofar %d  Total %d\n", i, active_term_buffer[i], active_term_buffer[i]->term->string, active_term_buffer[i]->tokens_so_far, active_term_buffer[i]->num_tokens);
        active_term_buffer[i]->tokens_so_far++;
    }

    // check if we've completed any of the active term(s)
    *num_completed_terms = 0;
    i = 0;
    while (i < *num_active_terms) {
        if (active_term_buffer[i]->tokens_so_far == active_term_buffer[i]->num_tokens) {
            // save this as a completed term
            completed_term_buffer[*num_completed_terms] = active_term_buffer[i];
            *num_completed_terms = *num_completed_terms + 1;
            // and pop it from the buffer
            ShiftActiveTermBuffer(active_term_buffer, i, *num_active_terms);
            *num_active_terms = *num_active_terms - 1;
        } else {
            i++;
        }
    }
}
void ProcessTermBuffers_VocabLookup(long long *tokens_since_last_annotation,
        struct term_annotation **passive_term_buffer, int *passive_term_buffer_ix,
        int buffer_size,
        struct term_annotation **active_term_buffer, int *num_active_terms,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        FILE *annothook, struct vocabulary *term_vocab) {
    ProcessTermBuffers_Master(tokens_since_last_annotation,
        passive_term_buffer, passive_term_buffer_ix, buffer_size, NULL,
        active_term_buffer, num_active_terms, completed_term_buffer,
        num_completed_terms, annothook, term_vocab);
}
//@}


/**
 * Move one slot forward in the word buffer (refilling it from the corpus
 * if necessary), and update term buffers.
 *
 * @returns 1 if there is more corpus to read, else 0
 */
//@{
int ParallelReadStep_Master(struct indexed_string **word_buffer,
        long long *word_byte_buffer, struct term_annotation **passive_term_buffer,
        long long *passive_term_byte_buffer, struct term_annotation **active_term_buffer,
        int buffer_size, int *word_buffer_ix, int words_to_keep,
        int *passive_term_buffer_ix, int *num_active_terms,
        long long *word_buffer_start_ix, long long *tokens_since_last_annotation,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        int max_string_len, FILE *plainhook, FILE *annothook,
        struct vocabulary *word_vocab, struct vocabulary *term_vocab) {

    // if we're at the end of the current word buffer, refill it
    if (*word_buffer_ix == (buffer_size-words_to_keep-1)) {
        ShiftWordBuffer_Master(word_buffer, word_byte_buffer, buffer_size,
            words_to_keep, plainhook, max_string_len, word_vocab);
        *word_buffer_ix = 0;   // reset the local counter
        *word_buffer_start_ix += (buffer_size-words_to_keep);   // update the global counter
    } else {
        *word_buffer_ix = *word_buffer_ix + 1;
    }

    // bump the annotation offset up
    *tokens_since_last_annotation = *tokens_since_last_annotation + 1;

    // process terms
    ProcessTermBuffers_Master(tokens_since_last_annotation,
        passive_term_buffer, passive_term_buffer_ix,
        buffer_size, passive_term_byte_buffer,
        active_term_buffer, num_active_terms,
        completed_term_buffer, num_completed_terms,
        annothook, term_vocab);

    // if we're finishing, check if we're done here
    if (word_buffer[*word_buffer_ix]->string[0] == 0) {
        return 0;  // no more to read
    } else {
        return 1;  // keep going!
    }
}
int ParallelReadStep_VocabLookup(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer, 
        struct term_annotation **active_term_buffer, int buffer_size,
        int *word_buffer_ix, int words_to_keep,
        int *passive_term_buffer_ix, int *num_active_terms,
        long long *word_buffer_start_ix, long long *tokens_since_last_annotation,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        int max_string_len, FILE *plainhook, FILE *annothook,
        struct vocabulary *word_vocab, struct vocabulary *term_vocab) {
    return ParallelReadStep_Master(word_buffer, NULL, passive_term_buffer, NULL,
        active_term_buffer, buffer_size, word_buffer_ix, words_to_keep,
        passive_term_buffer_ix, num_active_terms, word_buffer_start_ix,
        tokens_since_last_annotation, completed_term_buffer, num_completed_terms,
        max_string_len, plainhook, annothook, word_vocab, term_vocab);
}
int ParallelReadStep_ByteTracking(struct indexed_string **word_buffer,
        long long *word_byte_buffer,
        struct term_annotation **passive_term_buffer, long long *passive_term_byte_buffer,
        struct term_annotation **active_term_buffer, int buffer_size,
        int *word_buffer_ix, int words_to_keep,
        int *passive_term_buffer_ix, int *num_active_terms,
        long long *word_buffer_start_ix, long long *tokens_since_last_annotation,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        int max_string_len, FILE *plainhook, FILE *annothook) {
    return ParallelReadStep_Master(word_buffer, word_byte_buffer, passive_term_buffer,
        passive_term_byte_buffer, active_term_buffer, buffer_size, word_buffer_ix,
        words_to_keep, passive_term_buffer_ix, num_active_terms, word_buffer_start_ix,
        tokens_since_last_annotation, completed_term_buffer, num_completed_terms,
        max_string_len, plainhook, annothook, NULL, NULL);
}
int ParallelReadStep(struct indexed_string **word_buffer,
        struct term_annotation **passive_term_buffer,
        struct term_annotation **active_term_buffer, int buffer_size,
        int *word_buffer_ix, int words_to_keep,
        int *passive_term_buffer_ix, int *num_active_terms,
        long long *word_buffer_start_ix, long long *tokens_since_last_annotation,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        int max_string_len, FILE *plainhook, FILE *annothook) {
    return ParallelReadStep_Master(word_buffer, NULL, passive_term_buffer, NULL,
        active_term_buffer, buffer_size, word_buffer_ix, words_to_keep,
        passive_term_buffer_ix, num_active_terms, word_buffer_start_ix,
        tokens_since_last_annotation, completed_term_buffer, num_completed_terms,
        max_string_len, plainhook, annothook, NULL, NULL);
}
//@}
