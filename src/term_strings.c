#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include "vocab.h"
#include "io.h"
#include "logging.h"
#include "term_strings.h"

struct term_string_map *CreateTermStringMap(long num_terms) {
    long i;

    struct term_string_map *map = malloc(sizeof(struct term_string_map));
    if (map == NULL) {
        error("   >>> Failed to allocate memory for term string map; Aborting\n");
        exit(1);
    }

    map->strings = malloc(num_terms * sizeof(struct term_string));
    if (map->strings == NULL) {
        error("   >>> Failed to allocate memory for strings in term string map; Aborting\n");
        exit(1);
    }

    for (i = 0; i < num_terms; i++) {
        map->strings[i].tokens = NULL;
        map->strings[i].num_tokens = 0;
    }

    map->map_size = num_terms;

    return map;
}

void DestroyTermStringMap(struct term_string_map **map) {
    long i, j;

    if (*map != NULL) {
        if ((*map)->strings != NULL) {
            for (i = 0; i < (*map)->map_size; i++) {
                if ((*map)->strings[i].tokens != NULL) {
                    for (j = 0; j < (*map)->strings[i].num_tokens; j++) {
                        if ((*map)->strings[i].tokens[j] != NULL) {
                            free((*map)->strings[i].tokens[j]);
                            (*map)->strings[i].tokens[j] = NULL;
                        }
                    }
                    free((*map)->strings[i].tokens);
                    (*map)->strings[i].tokens = NULL;
                }
            }
            (*map)->map_size = 0;

            free((*map)->strings);
            (*map)->strings = NULL;
        }
        free(*map);
        *map = NULL;
    }
}

void ReadTermStringMap(char *mapfile, struct vocabulary *term_vocab, struct term_string_map *map) {
    long term_ix, found, i, j;
    char term_id[MAX_STRING];
    char ch;
    int len, num_tokens;
    char string[MAX_STRING * 100];
    char *word;

    FILE *f = fopen(mapfile, "rb");
    if (f == NULL) {
        error("Term->string map file %s not found\n", mapfile);
        exit(1);
    }

    info("  Reading term->string map from %s...\n", mapfile);
    found = 0;

    while (!feof(f)) {
        // first thing should be a term ID (integer)
        ReadWord(term_id, f, MAX_STRING);
        term_ix = SearchVocab(term_vocab, term_id);

        // now, pull characters out until we hit a newline
        len = 0;
        if (term_ix >= 0) {
            while (!feof(f)) {
                ch = fgetc(f);
                // tabs are only used to separate IDs from strings
                if (ch == '\t') {
                    if (len == 0) continue;
                    else {
                        error("   >>> Encountered unexpected TAB character in term-string map; Aborting\n");
                        exit(1);
                    }
                }
                // newlines end the string (some strings in the file may be empty)
                if (ch == '\n') break;
                // otherwise, just save the character
                string[len] = ch;
                if (len < MAX_STRING * 100)
                    len++;
            }
        } else {
            ch = fgetc(f);
            while (!feof(f) && ch != '\n') ch = fgetc(f);
        }
        string[len] = 0;

        // if the term is in the vocab, tokenize and save its string
        if (term_ix >= 0) {
            //printf("  [STRMAP] Read string \"%s\"\n", string);
            // first, go through and count the number of tokens in the string
            num_tokens = 1;
            for (i = 0; i < strlen(string); i++) {
                if (string[i] == ' ') num_tokens++;
            }
            map->strings[term_ix].num_tokens = num_tokens;
            //printf("  [STRMAP]    # tokens: %d\n", num_tokens);

            // now that we know how many tokens the string has, allocate memory
            // for their pointers
            map->strings[term_ix].tokens = malloc(num_tokens * sizeof(char *));
            if (map->strings[term_ix].tokens == NULL) {
                error("   >>> Failed to allocate memory for term token list; Aborting\n");
                exit(1);
            }

            // and go through and save each token
            //printf("  [STRMAP]    Tokenized: ");
            j = 0;
            word = strtok(string, " ");
            while (word != NULL) {
                map->strings[term_ix].tokens[j] = malloc((strlen(word) + 1) * sizeof(char));
                if (map->strings[term_ix].tokens[j] == NULL) {
                    error("   >>> Failed to allocate memory for token in term; Aborting\n");
                    exit(1);
                }
                strcpy(map->strings[term_ix].tokens[j], word);
                //printf("%s ", map->strings[term_ix].tokens[j]);
                word = strtok(NULL, " ");
                j++;
            }
            // make sure we read the right number of tokens
            if (j != num_tokens) {
                error("   >>> Found wrong number of tokens for term: %d (expected %d); Aborting\n", j, num_tokens);
                exit(1);
            }
            //printf("\n");

            found++;
        }
    }

    fclose(f);

    info("  Read strings for %ld terms (missing %ld).\n", found, map->map_size - found);
}
