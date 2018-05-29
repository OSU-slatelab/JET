// Modified by Denis Griffis, Jul-Aug 2016
// Added:
//   - Support for multiword string tags
//   
//
// Modifed by Yoav Goldberg, Jan-Feb 2014
// Removed:
//    hierarchical-softmax training
//    cbow
// Added:
//   - support for different vocabularies for words and contexts
//   - different input syntax
//
/////////////////////////////////////////////////////////////////
//
//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <unistd.h>
#include "logging.h"
#include "mem.h"
#include "mt19937ar.h"
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
#include "model_io.h"
#include "context_manager.h"
#include "pre_init.h"

#define MIN_QUEUE_SIZE 1000
#define MIN_QUEUE_WINDOWS 10
#define BUFFER_WINDOWS 3
#define MAX_TERM_WORDS 500
#define MAX_FILENAME_SIZE 10000

// plaintext and standoff annotations
char plaintext_corpus_file[MAX_FILENAME_SIZE], corpus_annotations_file[MAX_FILENAME_SIZE];

// files for saving vectors
char word_vectors_file[MAX_FILENAME_SIZE], term_vectors_file[MAX_FILENAME_SIZE], 
     entity_vectors_file[MAX_FILENAME_SIZE], context_vectors_file[MAX_FILENAME_SIZE];

// vocabulary files
char wvocab_file[MAX_FILENAME_SIZE];
char tvocab_file[MAX_FILENAME_SIZE];

// other files
char thread_config_file[MAX_FILENAME_SIZE];
char map_file[MAX_FILENAME_SIZE];

// saving settings
char param_file[MAX_FILENAME_SIZE];
struct hyperparameters params;

// configuration flags
int help = 0;
int binary = 0;
int debug_mode = 2;
int window = 5;
int min_count = 5;
int num_threads = 1;
int min_reduce = 1;
int use_position = 0;
int save_each_iter = 0;
int window_size = 5;
long long embedding_size = 100;
long random_seed = -1;
struct model_flags *flags;

long long corpus_token_count = 0;
real alpha = 0.025, starting_alpha, downsampling_rate = 0.00001;
long long alpha_schedule_interval = 10000;
real *word_embeddings, *term_embeddings, *entity_embeddings, *ctx_embeddings;
real *word_norms, *term_norms, *entity_norms, *ctx_norms;
int numiters = 5;
int word_burn_iters = 0;
char *str_map_sep;

// pre-initialization
char term_strmap_file[MAX_FILENAME_SIZE];

// progress tracking
long long word_count_all_threads = 0;
int curiter = 0;
bool training = false;

int *thread_pause_flags;
long long *thread_word_counts;
pthread_mutex_t saving_embeds;

struct vocabulary *wv;
struct vocabulary *ev;
struct vocabulary *tv;
struct entity_map *termmap;
struct term_string_map *strmap;
struct term_monogamy_map *monomap;

int negative = 15;
int *unitable;
real *word_downsampling_table;
real *term_downsampling_table;

/**
 * Set the input queue size to the larger of MIN_QUEUE_WINDOWS * window_size
 * and MIN_QUEUE_SIZE
 */
int GetQueueSize(int window_size) {
    int window_queue_slots = MIN_QUEUE_WINDOWS * window_size;
    if (window_queue_slots > MIN_QUEUE_SIZE)
        return window_queue_slots;
    else
        return MIN_QUEUE_SIZE;
}

void GetIterationFilename(char *basename, int iter, char *outname) {
    sprintf(outname, "%s.iter%d", basename, iter);
}


/**
 * Read an assigned chunk of the aligned untagged/tagged corpora,
 * and train all of word, term, and entity embeddings (as specified by user).
 * Contexts are words only.
 */
void *TrainModelThread(void *arguments) {
    struct thread_config *thread_args = arguments;

    // set up buffers for file reading
    int buffer_size = GetQueueSize(window_size);  // number of words/term annotations to store at a time
    struct indexed_string *word_buffer[buffer_size];
    struct term_annotation *passive_term_buffer[buffer_size];
    struct term_annotation *active_term_buffer[buffer_size];
    struct term_annotation *completed_term_buffer[buffer_size];
    AllocateBuffers(word_buffer, passive_term_buffer, buffer_size, MAX_STRING);

    // set up tracking info for file reading
    int num_active_terms = 0;
    int num_completed_terms = 0;
    long long word_buffer_start_ix = 0;
    int word_buffer_ix = 0;
    int passive_term_buffer_ix = 0;
    long long tokens_since_last_annotation = thread_args->start_offset_annot;

    int a, i;

    verbose("[Thread %d] -- Start byte (plaintext): %lld  Start byte (annotations): %lld  Start offset (annotated): %lld  # plaintext tokens: %lld\n", (int)thread_args->thread_id, thread_args->start_byte_plain, thread_args->start_byte_annot, thread_args->start_offset_annot, thread_args->tokens);

    // set up info for learning
    verbose("Initializing windows\n");
    int full_window_size = (window_size * 2) + 1;
    int target = window_size;
    int word_context_window[full_window_size];
    int masked_word_context_window[full_window_size];
    int max_num_entities = MaxNumEntities(termmap);

    int sampled_next_word_ix;
    int *sampled_completed_term_ixes = NULL;
    bool finishing_sentence;
    int sub_window_skip;
    int word_negative_samples[full_window_size * negative];
    int *term_negative_samples = NULL;
    int term_ns_block_start;

    // trackers for entity/context updates (since each can appear more
    // than once in a single update batch)
    // NOTE: these get initialized to 0 here, and LearningStep is
    // expected to reset all counters that it uses
    int *entity_update_counters = calloc(ev->vocab_size, sizeof(int));
    if (entity_update_counters == NULL) {
        error("   >>> Failed to allocate memory for entity update counters; Aborting\n");
        exit(1);
    }
    int *ctx_update_counters = calloc(wv->vocab_size, sizeof(int));
    if (ctx_update_counters == NULL) {
        error("   >>> Failed to allocate memory for context update counters; Aborting\n");
        exit(1);
    }

    int iter;
    long long corpus_token_count = wv->word_count;
    long long thread_word_count, last_report_word_count, last_alpha_word_count;
    bool halting;
    bool word_burn = false;


    // open up files
    FILE *plnhook = fopen(plaintext_corpus_file, "rb");
    FILE *annhook = fopen(corpus_annotations_file, "rb");
    if (plnhook == NULL) {
        error("Unable to read plaintext file %s\n", plaintext_corpus_file);
        exit(1);
    }
    if (annhook == NULL) {
        error("Unable to read annotations file %s\n", corpus_annotations_file);
        exit(1);
    }

    for (iter=0; iter < numiters; ++iter) {
        // before starting this iteration, wait until the progress tracker
        // says it's time to continue
        while (thread_pause_flags[thread_args->thread_id] == 1) {
            sleep(1);
        }

        // check if we're still in the burn-in iterations
        if (iter < word_burn_iters) word_burn = true;
        else word_burn = false;
        
        // start bytes are pre-calculated to put us at the (aligned) start of a word,
        // and not in the middle of a multi-word term
        fseek(plnhook, thread_args->start_byte_plain, SEEK_SET);
        fseek(annhook, thread_args->start_byte_annot, SEEK_SET);

        InitializeBuffersAndContexts(word_buffer, &word_buffer_ix,
            passive_term_buffer, &passive_term_buffer_ix,
            active_term_buffer, &num_active_terms,
            completed_term_buffer, &num_completed_terms,
            &sampled_next_word_ix, &sampled_completed_term_ixes,
            word_context_window, masked_word_context_window,
            &tokens_since_last_annotation, &finishing_sentence,
            buffer_size, full_window_size, window_size, target,
            word_downsampling_table, term_downsampling_table,
            plnhook, annhook, wv, tv);

        curiter = iter;
        finishing_sentence = false;
        halting = false;

        // no words processed yet for this iteration
        thread_word_counts[thread_args->thread_id] = 0;
        thread_word_count = 0;
        last_report_word_count = 0;
        last_alpha_word_count = 0;

        while (1) {

            //////////////////////////////////////
            // Progress tracking/management
            //////////////////////////////////////

            // pause if the embeddings are currently being saved
            pthread_mutex_lock(&saving_embeds);
            pthread_mutex_unlock(&saving_embeds);

            // check if we've processed the number of tokens allocated to this thread
            if (thread_word_count >= thread_args->tokens) {
            //if (thread_word_count >= 1000) {
                halting = true;
            }

            // update the master word count
            if (halting || thread_word_count - last_report_word_count > 100) {
                word_count_all_threads += thread_word_count - last_report_word_count;
                thread_word_counts[thread_args->thread_id] = thread_word_count;
                last_report_word_count = thread_word_count;
            }

            // handle alpha scheduling, based on number of words read
            if (alpha_schedule_interval > 0 && (thread_word_count - last_alpha_word_count > alpha_schedule_interval)) {
                alpha = starting_alpha * (1 - word_count_all_threads / (real)(numiters*corpus_token_count + 1));
                if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
                last_alpha_word_count = thread_word_count;
            }


            /////////////////////////////////////////
            // Sample additional info for learning
            // and update the model
            /////////////////////////////////////////

            if (!halting) {
                // choose a random subwindow size (to more highly weight closer
                // context words); sub_window_skip is the number of tokens to
                // REDUCE the window size by for this instance
                sub_window_skip = RandomSubwindowSkip((full_window_size - 1)/2);

                // get negative samples for the training word
                if (masked_word_context_window[target] >= 0) {
                    GetNegativeSamples(negative, word_negative_samples, 0, masked_word_context_window,
                        sub_window_skip, full_window_size - sub_window_skip, target, unitable);
                }
                // get negative samples for any training terms
                if (num_completed_terms > 0) {
                    term_negative_samples = malloc(num_completed_terms * full_window_size * negative * sizeof(int));
                    if (term_negative_samples == NULL) {
                        error("   >>> Failed to allocate memory for term negative samples; aborting\n");
                        exit(1);
                    }
                    for (i = 0; i < num_completed_terms; i++) {
                        if (sampled_completed_term_ixes[i] >= 0) {
                            term_ns_block_start = i * full_window_size * negative;
                            GetNegativeSamples(negative, term_negative_samples, term_ns_block_start,
                                completed_term_buffer[i]->contexts, sub_window_skip,
                                full_window_size - sub_window_skip, target, unitable);
                        }
                    }
                }

                LearningStep(masked_word_context_window, target, full_window_size, sub_window_skip,
                    completed_term_buffer, num_completed_terms, sampled_completed_term_ixes,
                    word_negative_samples, term_negative_samples, wv, tv, ev, termmap, monomap,
                    max_num_entities, word_embeddings, term_embeddings, entity_embeddings,
                    ctx_embeddings, word_norms, term_norms, entity_norms, ctx_norms,
                    entity_update_counters, ctx_update_counters,
                    alpha, embedding_size, negative, word_burn, flags);
            }


            //////////////////////////////////////
            // Memory cleanup before next step
            //////////////////////////////////////

            // clear out memory from downsampling terms
            if (sampled_completed_term_ixes != NULL) {
                free(sampled_completed_term_ixes);
                sampled_completed_term_ixes = NULL;
            }
            // clear out memory from term negative samples
            if (term_negative_samples != NULL) {
                free(term_negative_samples);
                term_negative_samples = NULL;
            }
            // clear out memory from completed terms that we just processed
            for (a = 0; a < num_completed_terms; a++) {
                // contexts
                if (completed_term_buffer[a]->contexts != NULL) {
                    free(completed_term_buffer[a]->contexts);
                    completed_term_buffer[a]->contexts = NULL;
                }
                // member words
                if (completed_term_buffer[a]->member_words != NULL) {
                    free(completed_term_buffer[a]->member_words);
                    completed_term_buffer[a]->member_words = NULL;
                }
            }

            // now that we've cleaned up, we can break the loop
            if (halting) {
                break;
            }

            ///////////////////////////////////////////
            // Step to next word and update contexts
            ///////////////////////////////////////////

            // step forward in the corpus
            ParallelReadStep_VocabLookup(word_buffer, passive_term_buffer, active_term_buffer,
                buffer_size, &word_buffer_ix, window_size+1, &passive_term_buffer_ix,
                &num_active_terms, &word_buffer_start_ix, &tokens_since_last_annotation,
                completed_term_buffer, &num_completed_terms, MAX_STRING, plnhook, annhook,
                wv, tv);

            // allocate memory for downsampling terms
            if (num_completed_terms > 0) {
                sampled_completed_term_ixes = malloc(num_completed_terms * sizeof(int));
                if (sampled_completed_term_ixes == NULL) {
                    error("   >>> Failed to allocate memory for completed_term_ixes\n");
                    exit(1);
                }
            }

            // downsample the next word and any completed terms
            DownsampleWordAndTerms(word_buffer, word_buffer_ix+window_size,
                completed_term_buffer, num_completed_terms, word_downsampling_table,
                term_downsampling_table, &sampled_next_word_ix, sampled_completed_term_ixes);

            // and update the context windows for the next learning step
            UpdateContextWindows(word_context_window, masked_word_context_window,
                sampled_next_word_ix, num_active_terms, active_term_buffer,
                num_completed_terms, completed_term_buffer, sampled_completed_term_ixes,
                &finishing_sentence, full_window_size, target);
            

            // bump the counters for progress tracking
            thread_word_count++;

        } // while loop - stepping over words (completed iteration)

        // flag that this iteration is complete
        thread_pause_flags[thread_args->thread_id] = 1;

    } // iterations loop
    fclose(plnhook);
    fclose(annhook);

    // clean up memory allocations
    DestroyBuffers(word_buffer, passive_term_buffer, buffer_size);

    pthread_exit(NULL);
}

/**
 * Save all the current model parameters
 */
void SaveModel(bool for_iter, int iter) {
    char this_file[MAX_STRING];

    // save context vectors
    if (context_vectors_file[0] != 0) {
        if (for_iter) sprintf(this_file, "%s.iter%d", context_vectors_file, iter);
        else strcpy(this_file, context_vectors_file);
        info("   Saving context embeddings to %s...", this_file);
        WriteVectors(this_file, wv, ctx_embeddings, embedding_size, binary);
        info("Done.\n");
    }

    // save word vectors
    if (word_vectors_file[0] != 0) {
        if (for_iter) sprintf(this_file, "%s.iter%d", word_vectors_file, iter);
        else strcpy(this_file, word_vectors_file);
        info("   Saving word embeddings to %s...", this_file);
        WriteVectors(this_file, wv, word_embeddings, embedding_size, binary);
        info("Done.\n");
    }

    // save term vectors
    if (term_vectors_file[0] != 0) {
        if (for_iter) sprintf(this_file, "%s.iter%d", term_vectors_file, iter);
        else strcpy(this_file, term_vectors_file);
        info("   Saving term embeddings to %s...", this_file);
        WriteVectors(this_file, tv, term_embeddings, embedding_size, binary);
        info("Done.\n");
    }

    // save entity vectors
    if (for_iter) sprintf(this_file, "%s.iter%d", entity_vectors_file, iter);
    else strcpy(this_file, entity_vectors_file);
    info("   Saving entity embeddings to %s...", this_file);
    WriteVectors(this_file, ev, entity_embeddings, embedding_size, binary);
    info("Done.\n");
}

/**
 * Print progress information as we iterate through the corpus
 */
void *TrackProgress(void *a) {
    long long last_word_count = 0, iter_word_count;
    int current_iteration = 0;
    int completed_threads, i;
    bool starting_training = true, changing_iteration = false, finishing_training = false;
    bool initialized_terms = false;

    time_t now, iter_start = time(NULL);
    int num_ticks, prog_bar_size = 30;
    char *progress_ticks = MallocOrDie(prog_bar_size+1, "progress bar ticks");
    progress_ticks[prog_bar_size] = 0;
    real progress;
    long long corpus_token_count = wv->word_count;
    while (!finishing_training) {

        if (!starting_training) {
            // check to see if all threads have hit the end of their
            // chunks for this iteration
            completed_threads = 0;
            for (i = 0; i < num_threads; i++)
                completed_threads += thread_pause_flags[i];
            if (completed_threads == num_threads) {
                changing_iteration = true;

                // check to see if we're done
                if (current_iteration == numiters)
                    finishing_training = true;
            } else {
                changing_iteration = false;
            }

            // print status of the current iteration
            if (changing_iteration || (word_count_all_threads - last_word_count) > 2000) {
                last_word_count = word_count_all_threads;

                // check how many words have been processed this iteration
                iter_word_count = 0;
                for (i = 0; i < num_threads; i++)
                    iter_word_count += thread_word_counts[i];

                if ((debug_mode > 1)) {
                    now = time(NULL);
                    if (changing_iteration) {
                        progress = 1.0;
                        num_ticks = prog_bar_size;
                    }
                    else {
                        progress = iter_word_count / (real)(corpus_token_count + 1);
                        num_ticks = (int)(progress * prog_bar_size);
                    }
                    for (i = 0; i < num_ticks; i++)
                        progress_ticks[i] = '>';
                    for (i = num_ticks; i < prog_bar_size; i++)
                        progress_ticks[i] = ' ';
                    info("\r Progress: %.1f%% [%s]  Words/thread/s: %.2fk Elapsed: %llds\n",
                        progress*100, progress_ticks,
                        (iter_word_count / (real)num_threads / (now - iter_start) / 1000),
                        now - iter_start
                    );
                }
            }
        }

        // and roll over to the next iteration if necessary
        if (!finishing_training && (starting_training || changing_iteration)) {
            if (save_each_iter && !starting_training) {
                // pause training to save the embedding state
                pthread_mutex_lock(&saving_embeds);
                info("\n\n");
                SaveModel(true, current_iteration);
                pthread_mutex_unlock(&saving_embeds);
            }

            current_iteration++;

            info("\n\nIteration %d/%d\n", current_iteration, numiters);
            iter_start = time(NULL);

            if (current_iteration > word_burn_iters && word_burn_iters > 0 && !initialized_terms) {
                info("  Initializing terms ... \n");
                InitTermsFromWords(term_embeddings, tv, word_embeddings, wv,
                    strmap, monomap, embedding_size);
                initialized_terms = true;
            }

            // flag that every thread is good to go again
            for (i = 0; i < num_threads; i++)
                thread_pause_flags[i] = 0;

            starting_training = false;
            changing_iteration = false;
        }
    }

    FreeAndNull((void *)&progress_ticks);
    pthread_exit(NULL);
}


/**
 * Master method for embedding training:
 *   (0) reads vocabularies and initializes the model
 *   (1) sets up all threads with the individual corpus chunks they read,
 *   (2) manages training thread execution and progress tracker thread
 */
void TrainModel(long long *thread_tokens, long long *thread_start_bytes_plain,
        long long *thread_start_bytes_annot, long long *thread_start_offsets_annot) {
    long a;
    pthread_t *pt = malloc((num_threads+1) * sizeof(pthread_t));
    if (pt == NULL) {
        error("   >>> Failed to allocate memory for training threads; Aborting\n");
        exit(1);
    }

    info("Starting training...\n");
    training = true;
    pthread_mutex_init(&saving_embeds, NULL);

    time_t full_training_start = time(NULL), full_training_stop;

    // start the training threads
    struct thread_config **thread_args = malloc(num_threads * sizeof(struct thread_config *));
    if (thread_args == NULL) {
        error("   >>> Failed to allocate memory for thread configurations; Aborting\n");
        exit(1);
    }
    for (a = 0; a < num_threads; a++) {
        thread_args[a] = malloc(sizeof(struct thread_config));
        if (thread_args[a] == NULL) {
            error("   >>> Failed to allocate memory for configuring thread %d; Aborting\n", a);
            exit(1);
        }
        thread_args[a]->thread_id = a;
        thread_args[a]->tokens = thread_tokens[a];
        thread_args[a]->start_byte_plain = thread_start_bytes_plain[a];
        thread_args[a]->start_byte_annot = thread_start_bytes_annot[a];
        thread_args[a]->start_offset_annot = thread_start_offsets_annot[a];
        pthread_create(&pt[a], NULL, TrainModelThread, (void *)thread_args[a]);
    }

    pthread_create(&pt[num_threads], NULL, TrackProgress, (void *)NULL);

    // wait for all training threads to complete
    for (a = 0; a < num_threads; a++) {
        pthread_join(pt[a], NULL);
    }

    // halt the progress tracking thread
    training = false;
    pthread_join(pt[num_threads], NULL);

    pthread_mutex_destroy(&saving_embeds);
    full_training_stop = time(NULL);
    info("\n\n\nTraining complete; total time elapsed: %.2fs.\n\n", (real)(full_training_stop - full_training_start));

    SaveModel(false, -1);

    // clean up memory for thread args
    if (thread_args != NULL) {
        for (a = 0; a < num_threads; a++) {
            if (thread_args[a] != NULL) {
                free(thread_args[a]);
                thread_args[a] = NULL;
            }
        }

        free(thread_args);
        thread_args = NULL;
    }
    // and for threads
    if (pt != NULL) {
        free(pt);
        pt = NULL;
    }
}

void LoadCorpusKnowledge() {
    long long full_corpus_size;
    bool save_word_vocab = false, save_term_vocab = false;

    // get word vocabulary (unfiltered; learn if necessary)
    info("Getting word vocabulary...\n");
    if (!FileExists(wvocab_file)) save_word_vocab = true;
    wv = GetWordVocabulary(plaintext_corpus_file, wvocab_file);

    // and save it if it wasn't already saved
    if (save_word_vocab && wvocab_file[0] != 0) {
        SortAndReduceVocab(wv, 1);
        SaveVocab(wv, wvocab_file);
        info("  Wrote word vocabulary to %s\n", wvocab_file);
    }

    // get term vocabulary (unfiltered; learn if necessary)
    info("Getting term vocabulary...\n");
    if (!FileExists(tvocab_file)) save_term_vocab = true;
    tv = GetTermVocabulary(corpus_annotations_file, tvocab_file);

    // and save it if it wasn't already saved
    if (save_term_vocab && tvocab_file[0] != 0) {
        SortAndReduceVocab(tv, 1);
        SaveVocab(tv, tvocab_file);
        info("  Wrote term vocabulary to %s\n", tvocab_file);
    }

    full_corpus_size = wv->word_count;

    // reduce and filter word/term vocabularies for active use
    SortAndReduceVocab(wv, min_count);
    SortAndReduceVocab(tv, min_count);
    info("  Filtered word vocabulary size: %ld (%lld tokens)\n", wv->vocab_size, wv->word_count);
    info("  Filtered term vocabulary size: %ld (%lld instances)\n", tv->vocab_size, tv->word_count);

    // the entity vocab will be generated while reading the term-entity map file
    ev = CreateVocabulary();
    // read the term-entity map
    info("Reading term->entity map...\n");
    termmap = ReadEntityMap(map_file, tv, ev, str_map_sep);

    info("  Filtered entity vocabulary size: %ld\n", ev->vocab_size);

    // read the term-string map
    info("Reading term->string map...\n");
    strmap = CreateTermStringMap(tv->vocab_size);
    ReadTermStringMap(term_strmap_file, tv, strmap);

    // calculate monogamy scores
    info("Calculating monogamy scores...\n");
    monomap = CreateMonogamyMap(tv->vocab_size);
    CalculateMonogamy(tv, wv, strmap, full_corpus_size, monomap);

    /*
    for (long term_ix = 0; term_ix < tv->vocab_size; term_ix++) {
        printf("  [MONOGAMY]  Term %ld \"", term_ix);
        for (int i = 0; i < strmap->strings[term_ix].num_tokens; i++) {
            printf("%s ", strmap->strings[term_ix].tokens[i]);
        }
        printf("\b\"  MW: %f\n", monomap->monogamies[term_ix].monogamy_weight);
        for (int j = 0; j < monomap->monogamies[term_ix].num_tokens; j++) {
            printf("  [MONOGAMY]     Word %d M: %f\n", j, monomap->monogamies[term_ix].by_word[j]);
        }
    }
    */
}

void header() {
    printf("                               ___\n");
    printf("                         |     | |\n");
    printf("                        / \\    | |\n");
    printf("    JET toolkit        |---|   |-|\n");
    printf("                       |---|   | |\n");
    printf(" Jointly embedding    /     \\  | |\n");
    printf(" Entities and Text   |       | | |\n");
    printf("                     | J     | | |\n");
    printf("   (c) The Ohio      |   E   | | |\n");
    printf("  State University   |     T | | |\n");
    printf("                     |_______| | |\n");
    printf("      GPL v3          |@| |@|  | |\n");
    printf("                      @@@ @@@  | | \n");
    printf("                     __________|_|_\n");
    printf("\n\n");

}

void usage() {
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-plaintext <file>\n");
    printf("\t\tUse plaintext corpus in <file> to train the model\n");
    printf("\t-annotations <file>\n");
    printf("\t\tUse annotations in <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting entity vectors\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 15, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-thread-config <file>\n");
    printf("\t\tWrite threading configuration to <file>; if <file> exists, read existing configuration from it\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words and contexts. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 1e-5");
    printf(" (smaller value means more aggressive downsampling).");
    printf(" A higher value for -sample corresponds to a higher threshold for downsampling.\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-alpha-schedule <int>\n");
    printf("\t\tSet the scheduling interval for decreasing the learning rate; 0 for no scheduling, default is 10,000 words\n");
    printf("\t-iters <int>\n");
    printf("\t\tPerform i iterations over the data; default is %d\n", numiters);
    printf("\t-word-burn-iters <int>\n");
    printf("\t\tOnly update words for the first <num> iterations; default is %d\n", word_burn_iters);
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-each <int>\n");
    printf("\t\tSave vectors at each iteration; default is 0 (off)\n");
    printf("\t-save-word-vectors <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors\n");
    printf("\t-save-context-vectors filename\n");
    printf("\t\tDump the context vectors in file <filename>\n");
    printf("\t-save-term-vectors filename\n");
    printf("\t\tDump the term vectors in file <filename>\n");
    printf("\t-word-vocab <file>\n");
    printf("\t\tFile to read word vocabulary from; if does not exist, word vocabulary will be learned and written to <file>\n");
    printf("\t--term-vocab <file>\n");
    printf("\t\tFile to read term vocabulary from; if does not exist, term vocabulary will be learned and written to <file>\n");
    printf("\t-save-entity-likelihoods <file>\n");
    printf("\t\tSave learned context-independent term-entity likelihoods to <file>\n");
    printf("\t-save-settings <file>\n");
    printf("\t\tSave learning settings to <file>\n");
    printf("\t-term-map filename\n");
    printf("\t\tfile mapping terms to entities\n");
    printf("\t-term-map-sep <char>\n");
    printf("\t\tcharacter separating entities in termmap file\n");
    printf("\t-initialize-from <file>\n");
    printf("\t\tFile containing pre-trained word embeddings to initialize the model from (requires -stringmap)\n");
    printf("\t-stringmap <file>\n");
    printf("\t\tFile mapping term IDs to the strings they represent (required if using -initialize-from)\n");
    printf("\nDEBUGGING OPTIONS\n");
    printf("\t-random-seed <seed>\n");
    printf("\t\tDebugging option; allows for a hard seed to the random number generator, for replicable behavior\n");
    printf("\t-disable-words\n");
    printf("\t\tDisables word training\n");
    printf("\t-disable-terms\n");
    printf("\t\tDisables term training\n");
    printf("\t-disable-entities\n");
    printf("\t\tDisables entity training\n");
    // TODO: fix example
    printf("\nExamples:\n");
    printf("./word2vecf -train data.txt -wvocab wv -cvocab ev -tvocab tv -output vec.txt -size 200 -negative 5 -threads 10 \n\n");
}

void parse_args(int argc, char **argv) {
    int i;

    plaintext_corpus_file[0] = 0;
    corpus_annotations_file[0] = 0;
    entity_vectors_file[0] = 0;
    wvocab_file[0] = 0;
    tvocab_file[0] = 0;
    map_file[0] = 0;
    word_vectors_file[0] = 0;
    context_vectors_file[0] = 0;
    term_vectors_file[0] = 0;
    thread_config_file[0] = 0;
    term_strmap_file[0] = 0;
    param_file[0] = 0;

    if ((i = ArgPos((char *)"-help", argc, argv)) > 0) help = 1;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) embedding_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-plaintext", argc, argv)) > 0) strcpy(plaintext_corpus_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-annotations", argc, argv)) > 0) strcpy(corpus_annotations_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-word-vocab", argc, argv)) > 0) strcpy(wvocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-term-vocab", argc, argv)) > 0) strcpy(tvocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-term-map", argc, argv)) > 0) strcpy(map_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha-schedule", argc, argv)) > 0) alpha_schedule_interval = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(entity_vectors_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-thread-config", argc, argv)) > 0) strcpy(thread_config_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) downsampling_rate = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-save-word-vectors", argc, argv)) > 0) strcpy(word_vectors_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-context-vectors", argc, argv)) > 0) strcpy(context_vectors_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-term-vectors", argc, argv)) > 0) strcpy(term_vectors_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-settings", argc, argv)) > 0) strcpy(param_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-iters", argc, argv)) > 0) numiters = atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-word-burn-iters", argc, argv)) > 0) word_burn_iters = atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-save-each", argc, argv)) > 0) save_each_iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-stringmap", argc, argv)) > 0) strcpy(term_strmap_file, argv[i + 1]);
    // debug options
    if ((i = ArgPos((char *)"-random-seed", argc, argv)) > 0) random_seed = atol(argv[i + 1]);
    if ((i = FlagPos((char *)"-disable-words", argc, argv)) > 0) flags->disable_words = true;
    if ((i = FlagPos((char *)"-disable-terms", argc, argv)) > 0) flags->disable_terms = true;
    if ((i = FlagPos((char *)"-disable-entities", argc, argv)) > 0) flags->disable_entities = true;

    // handle default setting for termmap-sep
    if ((i = ArgPos((char *)"-term-map-sep", argc, argv)) > 0) str_map_sep = argv[i + 1];
    else {
        str_map_sep = malloc(2 * sizeof(char));
        strcpy(str_map_sep, ",");
    }
}

int verify_args() {
    // check for required files
    if (plaintext_corpus_file[0] == 0) { printf("must supply -plaintext.\n\n"); return 0; }
    if (corpus_annotations_file[0] == 0) { printf("must supply -annotations.\n\n"); return 0; }
    if (entity_vectors_file[0] == 0) { printf("must supply -output.\n\n"); return 0; }
    if (map_file[0] == 0) { printf("must supply -term-map.\n\n"); return 0; }

    // burn-in check
    if (word_burn_iters >= numiters) {
        printf("-word-burn-iters must be less than -iters.\n\n");
        return 0;
    }

    // validate hyperparameters
    if (window_size <= 0) {
        printf("-window-size must be greater than 0.\n\n");
        return 0;
    }
    if (embedding_size <= 0) {
        printf("-size must be greater than 0.\n\n");
        return 0;
    }
    if (alpha <= 0) {
        printf("-alpha must be greater than 0.\n\n");
        return 0;
    }
    if (alpha_schedule_interval <= 0) {
        printf("-alpha-schedule must be greater than 0.\n\n");
        return 0;
    }
    if (negative < 0) {
        printf("-negative must be 0 or greater.\n\n");
        return 0;
    }
    if (numiters <= 0) {
        printf("-iters must be greater than 0.\n\n");
        return 0;
    }
    if (word_burn_iters < 0) {
        printf("-word-burn-iters must be 0 or greater.\n\n");
        return 0;
    }

    return 1;
}

int main(int argc, char **argv) {
    set_log_level(INFO);
    //set_log_level(VERBOSE);
    header();
    
    // set up the model flags (will be set from command line arguments)
    InitModelFlags(&flags);

    // handle command line arguments
    if (argc == 1) {
        usage();
        return 0;
    }
    parse_args(argc, argv);
    if (help == 1) {
        usage();
        return 0;
    }
    if (!verify_args()) { return 0; }

    // give notice of any disabled model components
    if (flags->disable_words
            || flags->disable_terms
            || flags->disable_entities) {
        info("=== Model overrides ===\n");
        if (flags->disable_words)
            info("  Word training: DISABLED\n");
        if (flags->disable_terms)
            info("  Term training: DISABLED\n");
        if (flags->disable_entities)
            info("  Entity training: DISABLED\n");
        info("\n");
    }

    // seed the random number generator
    if (random_seed <= 0)
        random_seed = (long)time(NULL);
    init_genrand(random_seed);

    // write model settings
    params.plaintext_corpus_file = plaintext_corpus_file;
    params.corpus_annotations_file = corpus_annotations_file;
    params.numiters = numiters;
    params.word_burn_iters = word_burn_iters;
    params.window = window_size;
    params.min_count = min_count;
    params.embedding_size = embedding_size;
    params.alpha = alpha;
    params.alpha_schedule_interval = alpha_schedule_interval;
    params.downsampling_rate = downsampling_rate;
    params.random_seed = random_seed;
    params.flags = flags;
    params.num_threads = num_threads;
    params.map_file = map_file;
    params.str_map_sep = str_map_sep;
    params.term_strmap_file = term_strmap_file;
    params.thread_config_file = thread_config_file;
    params.wvocab_file = wvocab_file;
    params.tvocab_file = tvocab_file;
    params.word_vectors_file = word_vectors_file;
    params.term_vectors_file = term_vectors_file;
    params.entity_vectors_file = entity_vectors_file;
    params.context_vectors_file = context_vectors_file;
    // TODO: make this required
    if (param_file[0] != 0) {
        WriteHyperparameters(param_file, params);
        info("Wrote current configuration to %s.\n\n", param_file);
    }

    time_t overall_start = time(NULL), overall_stop;

    // get the threading configuration
    long long *thread_tokens, *thread_start_bytes_plain,
        *thread_start_bytes_annot, *thread_start_offsets_annot;
    AllocateThreadConfigurations(&thread_tokens, &thread_start_bytes_plain,
        &thread_start_bytes_annot, &thread_start_offsets_annot, num_threads);

    ThreadConfigureMaster(thread_config_file, plaintext_corpus_file,
        corpus_annotations_file, thread_tokens, thread_start_bytes_plain,
        thread_start_bytes_annot, thread_start_offsets_annot, num_threads, false);

    // set up shared memory for threaded progress tracking
    thread_pause_flags = calloc(num_threads, sizeof(int));
    if (thread_pause_flags == NULL) {
        error("   >>> Failed to allocate memory for thread pause flags; Aborting\n");
        exit(1);
    }
    thread_word_counts = calloc(num_threads, sizeof(long long));
    if (thread_word_counts == NULL) {
        error("   >>> Failed to allocate memory for thread word counts; Aborting\n");
        exit(1);
    }

    // grab vocabularies, term-entity map, term-string map, etc.
    LoadCorpusKnowledge();

    info("Learning vectors using plaintext file %s\n", plaintext_corpus_file);
    info("                     and annotations file %s\n", corpus_annotations_file);
    
    starting_alpha = alpha;

    info("\nInitializing model...\n");
    InitializeModel(&word_embeddings, &term_embeddings, &entity_embeddings, &ctx_embeddings,
        &word_norms, &term_norms, &entity_norms, &ctx_norms,
        wv, tv, ev, termmap, embedding_size, &unitable, &word_downsampling_table,
        &term_downsampling_table, downsampling_rate);

    TrainModel(thread_tokens, thread_start_bytes_plain, thread_start_bytes_annot,
        thread_start_offsets_annot);

    overall_stop = time(NULL);
    info("\nProcessing complete.\n  >> Total time elapsed: %.2fs.\n", (real)(overall_stop - overall_start));

    // clean up memory
    DestroyEntityMap(&termmap);
    DestroyVocabulary(&wv);
    DestroyVocabulary(&tv);
    DestroyVocabulary(&ev);
    DestroyTermStringMap(&strmap);
    DestroyMonogamyMap(&monomap);
    DestroyModel(&word_embeddings, &term_embeddings, &entity_embeddings, &ctx_embeddings,
        &word_norms, &term_norms, &entity_norms, &ctx_norms,
        &unitable, &word_downsampling_table, &term_downsampling_table);
    DestroyThreadConfigurations(&thread_tokens, &thread_start_bytes_plain,
        &thread_start_bytes_annot, &thread_start_offsets_annot);
    DestroyModelFlags(&flags);

    if (thread_pause_flags != NULL) {
        free(thread_pause_flags);
        thread_pause_flags = NULL;
    }
    if (thread_word_counts != NULL) {
        free(thread_word_counts);
        thread_word_counts = NULL;
    }

    return 0;
}
