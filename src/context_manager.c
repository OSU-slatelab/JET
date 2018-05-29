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
#include "term_strings.h"
#include "model.h"
#include "context_manager.h"

/**
 * Move all long integers in the given window num_elements to the left.
 * Sets emptied slots to -1.
 */
void ShiftIntWindow(int *window, int num_elements, int window_size) {
    int i;
    // shift over num_elements slots
    for (i = num_elements; i < window_size; i++) {
        window[i-num_elements] = window[i];
    }
    // mark the remaining entries as empty
    for (i = window_size-num_elements; i < window_size; i++) {
        window[i] = -1;
    }
}

/**
 * Randomly choose to use or ignore (downsample) the next word and any
 * completed terms, based on their frequency.  (Higher frequency words/
 * terms are more likely to be downsampled.)
 *
 * Sampled word and term indices are assigned to the input pointers.
 */
void DownsampleWordAndTerms(struct indexed_string **word_buffer, int next_word_ix,
        struct term_annotation **completed_term_buffer, int num_completed_terms,
        real *word_downsampling_table, real *term_downsampling_table,
        int *sampled_next_word_ix, int *sampled_completed_term_ixes) {
    int sampling_ix;

    // check for downsampling the next word
    sampling_ix = word_buffer[next_word_ix]->vocab_index;
    if (sampling_ix > 0 && RollToDownsample(word_downsampling_table, sampling_ix)) {
        *sampled_next_word_ix = -1;
    } else {
        *sampled_next_word_ix = sampling_ix;
    }

    // check for downsampling each completed_term
    for (int i = 0; i < num_completed_terms; i++) {
        sampling_ix = completed_term_buffer[i]->term->vocab_index;
        if (sampling_ix >= 0 && RollToDownsample(term_downsampling_table, sampling_ix)) {
            sampled_completed_term_ixes[i] = -1;
        } else {
            sampled_completed_term_ixes[i] = sampling_ix;
        }
    }
}

/**
 * Update the contents of word and term context windows after a step
 * to the next corpus word.
 *
 * Word context window gets shifted left by 1, and the next (sampled)
 * word index is put into the last slot.
 *
 * Term context windows are determined in two phases:
 *   (1) Left side context is saved when a term begins
 *   (2) Right side context is saved when a term completes
 */
void UpdateContextWindows(int *word_context_window, int *masked_word_context_window,
        int sampled_next_word_ix, int num_active_terms, struct term_annotation **active_term_buffer,
        int num_completed_terms, struct term_annotation **completed_term_buffer,
        int *sampled_completed_term_ixes, bool *finishing_sentence, int full_window_size,
        int target) {
    int window_size = target;
    int a, b, mask_ix;

    // shift the context window left
    ShiftIntWindow(word_context_window, 1, full_window_size);
    // and put the next word into it (may already be downsampled, agnostic here)
    word_context_window[full_window_size-1] = sampled_next_word_ix;
    // update the masked context window (masked right of sentence terminator)
    mask_ix = -1;
    for (a = 0; a < full_window_size; a++) {
        if (mask_ix != -1) masked_word_context_window[a] = -1;
        else {
            if (a >= target && word_context_window[a] == 0) mask_ix = a;
            masked_word_context_window[a] = word_context_window[a];
        }
    }

    // if we just pulled a sentence termination token into the window, mark it
    if (word_context_window[full_window_size-1] == 0)
        *finishing_sentence = true;
    // but if we just finished a sentence, flip the switch and erase the left contexts
    if (finishing_sentence && word_context_window[target] == 0) {
        *finishing_sentence = false;
        for (a = 0; a < target; a++) word_context_window[a] = -1;
    }


    /////////////////////////////////////
    // Manage term contexts and words
    /////////////////////////////////////

    // handle all active terms
    for (a = 0; a < num_active_terms; a++) {
        // any term with NULL contexts is a newly-activated one, so copy
        // in the left contexts of the current word as their left contexts
        if (active_term_buffer[a]->contexts == NULL) {
            active_term_buffer[a]->contexts = malloc(full_window_size * sizeof(int));
            if (!active_term_buffer[a]->contexts) {
                error("   >>> Failed to allocate memory for contexts for a new active term\n");
                exit(1);
            }
            for (b = 0; b < window_size; b++) {
                active_term_buffer[a]->contexts[b] = word_context_window[b];
            }
        }
        // similarly, if it has no component words yet, start tracking those
        if (active_term_buffer[a]->member_words == NULL) {
            active_term_buffer[a]->member_words =
                malloc(active_term_buffer[a]->num_tokens * sizeof(int));
            if (!active_term_buffer[a]->member_words) {
                error("   >>> Failed to allocate memory for member words for a new active term\n");
                exit(1);
            }
        }
        // finally, mark the current word in EVERY active term
        active_term_buffer[a]->member_words[active_term_buffer[a]->tokens_so_far-1] =
            word_context_window[target];
    }
    // handle newly-completed terms
    for (a = 0; a < num_completed_terms; a++) {
        // if the term has been downsampled, it won't be used, so we can
        // ignore its contexts here
        if (sampled_completed_term_ixes[a] >= 0) {
            // if the term has no prior contexts, it's a single word term,
            // so copy in the current left contexts as well
            if (completed_term_buffer[a]->contexts == NULL) {
                completed_term_buffer[a]->contexts = MallocOrDie(full_window_size * sizeof(int), "completed term buffer contexts");
                for (b = 0; b < window_size; b++) {
                    completed_term_buffer[a]->contexts[b] = word_context_window[b];
                }
            }
            // now copy in the current right contexts for all completed terms
            for (b = target+1; b < full_window_size; b++) {
                completed_term_buffer[a]->contexts[b] = masked_word_context_window[b];
            }
            // if the term has no prior member words, it's a single word term,
            // so allocate memory here
            if (completed_term_buffer[a]->member_words == NULL) {
                completed_term_buffer[a]->member_words = malloc(1 * sizeof(int));
            }
            // finally, mark the current word in every completed term
            completed_term_buffer[a]->member_words[completed_term_buffer[a]->num_tokens-1] =
                word_context_window[target];
        }
    }
}

/**
 * At the start of corpus processing, fill the word and term buffers
 * and initialize all context windows.
 */
void InitializeBuffersAndContexts(struct indexed_string **word_buffer, int *word_buffer_ix,
        struct term_annotation **passive_term_buffer, int *passive_term_buffer_ix,
        struct term_annotation **active_term_buffer, int *num_active_terms,
        struct term_annotation **completed_term_buffer, int *num_completed_terms,
        int *sampled_next_word_ix, int **sampled_completed_term_ixes,
        int *word_context_window, int *masked_word_context_window,
        long long *tokens_since_last_annotation, bool *finishing_sentence, int buffer_size,
        int full_window_size, int window_size, int target, real *word_downsampling_table,
        real *term_downsampling_table, FILE *plnhook, FILE *annhook,
        struct vocabulary *word_vocab, struct vocabulary *term_vocab) {

    int sampling_ix, a;

    // load up the word and term buffers, getting the vocabulary indices of each
    // word and term at read time
    PreloadBuffers_VocabLookup(word_buffer, passive_term_buffer, buffer_size,
        plnhook, annhook, MAX_STRING, word_vocab, term_vocab);
    // process any initially-active terms
    ProcessTermBuffers_VocabLookup(tokens_since_last_annotation,
        passive_term_buffer, passive_term_buffer_ix, buffer_size,
        active_term_buffer, num_active_terms,
        completed_term_buffer, num_completed_terms,
        annhook, term_vocab);

    // set the current word context window to empty
    for (a = 0; a < full_window_size; a++)
        word_context_window[a] = -1;
    // load up the right side of the current word context window
    for (a = 0; a < window_size; a++) {
        sampling_ix = word_buffer[a]->vocab_index;
        if (sampling_ix > 0 && RollToDownsample(word_downsampling_table, sampling_ix)) {
            word_context_window[target + 1 + a] = -1;
        } else {
            word_context_window[target + 1 + a] = sampling_ix;
        }
    }

    // allocate memory for any completed terms
    if (num_completed_terms > 0) {
        *sampled_completed_term_ixes = malloc(*num_completed_terms * sizeof(int));
        if (*sampled_completed_term_ixes == NULL) {
            error("   >>> Failed to allocate memory for completed term ixes\n");
            exit(1);
        }
    }

    // now, optionally downsample the next word and any terms completed on the
    // first word
    DownsampleWordAndTerms(word_buffer, *word_buffer_ix+window_size,
        completed_term_buffer, *num_completed_terms, word_downsampling_table,
        term_downsampling_table, sampled_next_word_ix, *sampled_completed_term_ixes);

    // and then shift it to focus on the target word, and update
    // word/term contexts accordingly
    UpdateContextWindows(word_context_window, masked_word_context_window,
        *sampled_next_word_ix, *num_active_terms, active_term_buffer,
        *num_completed_terms, completed_term_buffer, *sampled_completed_term_ixes,
        finishing_sentence, full_window_size, target);
}
