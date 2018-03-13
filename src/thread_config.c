#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <limits.h>
#include "io.h"
#include "vocab.h"
#include "logging.h"
#include "parallel_reader.h"


/**
 * Configures token-wise start positions for each thread in both
 * the plaintext corpus and the standoff annotations file.
 *
 * @input plaintextf The plaintext corpus file
 * @input annotationsf The standoff annotations file
 * @input thread_tokens The number of plaintext tokens to be read by each thread
 *                      (initially empty)
 * @input thread_starts_untagged
 */
void ConfigureThreads(char *plaintextf, char *annotationsf, long long *thread_tokens, 
        long long *thread_start_bytes_plain, long long *thread_start_bytes_annot, 
        long long *thread_start_offsets_annot, int nthread) {

    // set up buffers
    int buffer_size = 10000;  // number of words/term annotations to store at a time
    struct indexed_string *word_buffer[buffer_size];
    long long word_byte_buffer[buffer_size];
    struct term_annotation *passive_term_buffer[buffer_size];
    long long term_byte_buffer[buffer_size];
    struct term_annotation *active_term_buffer[buffer_size];
    struct term_annotation *completed_term_buffer[buffer_size];
    AllocateBuffers(word_buffer, passive_term_buffer, buffer_size, MAX_STRING);

    // set up tracking info
    int num_active_terms = 0;
    int num_completed_terms = 0;
    long long word_buffer_start_ix = 0;
    int word_buffer_ix = 0;
    int passive_term_buffer_ix = 0;
    long long tokens_since_last_annotation = 0;

    // Calculate approximate byte size for each thread to process
    long long pln_size, pln_thread_size;
    double progress;
    pln_size = GetFileSize(plaintextf);
    pln_thread_size = (long long)( (pln_size / nthread) + 1 ); // ensure the last thread is always
                                                               // shorted by a few bytes (for tracking
                                                               // the number of tokens)

    // per-thread information
    int cur_thread = 0;
    long long pln_thread_ctr = 0;

    // open up files and fill the buffers
    FILE *plainhook = fopen(plaintextf, "rb");
    FILE *annothook = fopen(annotationsf, "rb");
    PreloadBuffers_ByteTracking(word_buffer, word_byte_buffer, passive_term_buffer, term_byte_buffer,
        buffer_size, plainhook, annothook, MAX_STRING);

    // set the initial start byte indices to 0
    thread_start_bytes_plain[0] = 0;
    thread_start_bytes_annot[0] = 0;
    // and mark an initial annotation offset of 0 for the first thread
    thread_start_offsets_annot[0] = 0;

    // loop over every word in the untagged document
    long long current_corpus_ix = 0;
    int continue_reading = 1;
    while (continue_reading == 1) {
        
        current_corpus_ix = word_buffer_start_ix + word_buffer_ix;
        debug("Word: %s  At byte: %lld\n", word_buffer[word_buffer_ix]->string, word_byte_buffer[word_buffer_ix]);

        // if we've run over the size for this thread and we're not in a term,
        // save the bounds and start tracking for the next thread
        if (
                num_active_terms == 0
                &&
                word_byte_buffer[word_buffer_ix] > ( (cur_thread + 1) * pln_thread_size)
           ) {
            // save number of tokens for this thread
            thread_tokens[cur_thread] = pln_thread_ctr + 1;
            // and the start bounds for the next thread
            if (cur_thread < nthread-1) {
                // byte offsets -- take the offset of the NEXT word, because the current word
                // is one we've already read and processed
                thread_start_bytes_plain[cur_thread+1] = word_byte_buffer[word_buffer_ix + 1];
                // but the next term in the passive term buffer is yet to be read, so use it
                thread_start_bytes_annot[cur_thread+1] = term_byte_buffer[passive_term_buffer_ix];
                // annotation offset -- because starting with the NEXT word, bump up the token
                // counter by one
                thread_start_offsets_annot[cur_thread+1] = tokens_since_last_annotation + 1;
            }
            // reset threaded trackers
            pln_thread_ctr = 0;
            cur_thread++;
        }
        else {
            pln_thread_ctr++;
        }

        if (current_corpus_ix % 1000000 == 0) {
            progress = (word_byte_buffer[word_buffer_ix] / (double)pln_size) * 100;
            info("%c  >> Processed (%2d%): %lld tokens", 13,
                (int)progress, current_corpus_ix);
        }

        continue_reading = ParallelReadStep_ByteTracking(word_buffer, word_byte_buffer,
            passive_term_buffer, term_byte_buffer, active_term_buffer,
            buffer_size, &word_buffer_ix, 1, &passive_term_buffer_ix, &num_active_terms,
            &word_buffer_start_ix, &tokens_since_last_annotation, completed_term_buffer,
            &num_completed_terms, MAX_STRING, plainhook, annothook);
    }
    info("\n");

    // save the number of tokens in the last thread (will be over
    // by 1 because of reading the EOF token, so decrement)
    thread_tokens[nthread-1] = pln_thread_ctr - 1;

    // cleanup
    DestroyBuffers(word_buffer, passive_term_buffer, buffer_size);
    fclose(plainhook);
    fclose(annothook);
}

/**
 * Write a threading configuration (# of plaintext tokens, plaintext starting byte offset,
 * annotations starting byte offset; for each thread) to a file
 */
void WriteThreadConfiguration(char *configf, long long *thread_tokens,
        long long *thread_start_bytes_plain, long long *thread_start_bytes_annot,
        long long *thread_start_offsets_annot, int nthread) {
    FILE *fo = fopen(configf, "wb");
    // error check
    if (fo == NULL) {
        error("Unable to write threading configuration to %s, does the directory exist?\n", configf);
        exit(1);
    }
    // header: # threads
    fprintf(fo, "%d\n", nthread);
    // # tokens, untagged byte, tagged byte for each thread
    for (int i=0; i<nthread; i++)
        fprintf(fo, "%lld %lld %lld %lld\n", thread_tokens[i], thread_start_bytes_plain[i],
                thread_start_bytes_annot[i], thread_start_offsets_annot[i]);
    fclose(fo);
}

/**
 * Read a previously-saved threading configuration from a file
 */
void ReadThreadConfiguration(char *configf, long long *thread_tokens,
        long long *thread_start_bytes_plain, long long *thread_start_bytes_annot,
        long long *thread_start_offsets_annot, int nthread) {
    int read_nthread = -1;
    FILE *fin = fopen(configf, "rb");

    // error check
    if (fin == NULL) {
        error("Unable to read threading configuration from %s\n", configf);
        exit(1);
    }

    // check that this configuration file is for the right number of threads
    fscanf(fin, "%d", &read_nthread);
    if (nthread != read_nthread) {
        error("Threading configuration file %s has thread number mismatch:\n", configf);
        error("  Configured for: %d threads\n", read_nthread);
        error("  Invoked with: %d threads\n", nthread);
        exit(1);
    }

    // read in the thread configurations
    for (int i=0; i<nthread; i++)
        fscanf(fin, "%lld %lld %lld %lld", &thread_tokens[i], &thread_start_bytes_plain[i],
               &thread_start_bytes_annot[i], &thread_start_offsets_annot[i]);

    verbose("Read thread configuration:\n");
    verbose("  # of threads: %d\n", read_nthread);
    for (int i=0; i<nthread; i++) {
        verbose("  Thread %d --", i);
        verbose("   # of tokens: %lld", thread_tokens[i]);
        verbose("   Start byte (plaintext): %lld", thread_start_bytes_plain[i]);
        verbose("   Start byte (annotations): %lld", thread_start_bytes_annot[i]);
        verbose("   Start offset (annotations): %lld", thread_start_offsets_annot[i]);
        verbose("\n");
    }
}

/**
 * All-in-one method; if configuration exists, read it
 * If it does not exist, configure threads and optionally save to file
 */
void ThreadConfigureMaster(char *thread_config_file, char *train_file_plain,
        char *train_file_annot, long long *thread_tokens,
        long long *thread_start_bytes_plain, long long *thread_start_bytes_annot,
        long long *thread_start_offsets_annot, int num_threads, bool force) {
    if (thread_config_file != 0 && !force && FileExists(thread_config_file)) {
        info("Reading existing threading configuration from %s...\n", thread_config_file);
        ReadThreadConfiguration(thread_config_file, thread_tokens, thread_start_bytes_plain,
            thread_start_bytes_annot, thread_start_offsets_annot, num_threads);
    } else {
        info("Calculating threading configuration...\n");
        ConfigureThreads(train_file_plain, train_file_annot, thread_tokens, 
            thread_start_bytes_plain, thread_start_bytes_annot, thread_start_offsets_annot, num_threads);
        if (thread_config_file[0] != 0) {
            info("Saving threading configuration to %s...\n", thread_config_file);
            WriteThreadConfiguration(thread_config_file, thread_tokens, thread_start_bytes_plain, 
                    thread_start_bytes_annot, thread_start_offsets_annot, num_threads);
        }
    }
}

/**
 * Allocate memory for thread configuration information
 */
void AllocateThreadConfigurations(long long **thread_tokens,
        long long **thread_start_bytes_plain, long long **thread_start_bytes_annot,
        long long **thread_start_offsets_annot, int num_threads) {
    *thread_tokens = malloc(num_threads * sizeof(long long));
    if (*thread_tokens == NULL) {
        error("   >>> Failed to allocate memory for thread configuration (tokens); Aborting\n");
        exit(1);
    }
    *thread_start_bytes_plain = malloc(num_threads * sizeof(long long));
    if (*thread_start_bytes_plain == NULL) {
        error("   >>> Failed to allocate memory for thread configuration (plaintext start byte); Aborting\n");
        exit(1);
    }
    *thread_start_bytes_annot = malloc(num_threads * sizeof(long long));
    if (*thread_start_bytes_annot == NULL) {
        error("   >>> Failed to allocate memory for thread configuration (annotations start byte); Aborting\n");
        exit(1);
    }
    *thread_start_offsets_annot = malloc(num_threads * sizeof(long long));
    if (*thread_start_offsets_annot == NULL) {
        error("   >>> Failed to allocate memory for thread configuration (annotations start offset); Aborting\n");
        exit(1);
    }
}

/**
 * Destroy memory allocations for thread configuration information
 */
void DestroyThreadConfigurations(long long **thread_tokens,
        long long **thread_start_bytes_plain, long long **thread_start_bytes_annot,
        long long **thread_start_offsets_annot) {
    if (*thread_tokens != NULL) {
        free(*thread_tokens);
        *thread_tokens = NULL;
    }
    if (*thread_start_bytes_plain) {
        free(*thread_start_bytes_plain);
        *thread_start_bytes_plain = NULL;
    }
    if (*thread_start_bytes_annot) {
        free(*thread_start_bytes_annot);
        *thread_start_bytes_annot = NULL;
    }
    if (*thread_start_offsets_annot) {
        free(*thread_start_offsets_annot);
        *thread_start_offsets_annot = NULL;
    }
}
