#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include "vocab.h"
#include "entities.h"
#include "io.h"
#include "logging.h"

const int term_hash_size = 50000000;  // Maximum 50M unique terms in the vocabulary

/// Term-Entity map handling //////////////////////////

struct entity_map *CreateEntityMap() {
    struct entity_map *m = malloc(sizeof(struct entity_map));
    if (!m) {
        error("   >>> Failed to allocate memory for entity map container\n");
        exit(1);
    }

    long long a;
    m->map_max_size = 1000;
    m->map_size = 0;

    m->map = (struct term_entities *)calloc(m->map_max_size, sizeof(struct term_entities));
    if (!m->map) {
        error("   >>> Failed to allocate memory for entity map\n");
        exit(1);
    }
    // explicitly set all entities pointers to NULL
    for (int i = 0; i < m->map_max_size; i++) {
        m->map[i].entities = NULL;
    }

    m->term_hash = (int *)calloc(term_hash_size, sizeof(int));
    if (!m->term_hash) {
        error("   >>> Failed to allocate memory for entity map term hash\n");
        exit(1);
    }
    for (a = 0; a < term_hash_size; a++) m->term_hash[a] = -1;

    return m;
}

/**
 * Free and set to NULL all memory allocated for this entity_map
 */
void DestroyEntityMap(struct entity_map **m) {
    int i;
    if (*m) {
        if ((*m)->map) {
            for (i = 0; i < (*m)->map_size; i++) {
                // only entities is allocated within the map;
                // the term pointer is handled in a vocabulary
                if ((*m)->map[i].entities) {
                    free((*m)->map[i].entities);
                    (*m)->map[i].entities = NULL;
                }
            }

            free((*m)->map);
            (*m)->map = NULL;
        }
        if ((*m)->term_hash) {
            free((*m)->term_hash);
            (*m)->term_hash = NULL;
        }

        free(*m);
        *m = NULL;  // dereference the map itself
    }
}

int CountEntities(char *entities, char *sep) {
    // count up the number of commas in the comma-separated string
    int i, commas=0;
    for (i=0; i < strlen(entities); i++) {
        if (entities[i] == *sep) commas++;
    }
    return commas+1;
}

void SplitEntities(char *entities, int num_entities, char **splits, char *sep) {
    char *split = strtok(entities, (const char *)sep);

    // copy each entity string in
    for (int i=0; i < num_entities; i++) {
        strcpy(splits[i], split);
        split = strtok(NULL, sep);  // continue splitting same source string
    }
}

int AddTermToMap(struct entity_map *m, char *term_string, char *entities,
        struct vocabulary *term_vocab, struct vocabulary *entity_vocab,
        char *sep) {
    int i;
    int num_entities = CountEntities(entities, sep);

    // allocate memory for the entity strings (not re-used elsewhere)
    char **entity_ids = malloc(sizeof(char*) * num_entities);
    if (!entity_ids) {
        error("   >>> Failed to allocate memory for parsing entity IDs\n");
        exit(1);
    }
    for (i = 0; i < num_entities; i++) {
        entity_ids[i] = malloc(sizeof(char) * MAX_STRING);
        if (!entity_ids[i]) {
            error("   >>> Failed to allocate memory for parsing single entity ID\n");
            exit(1);
        }
    }

    SplitEntities(entities, num_entities, entity_ids, sep);

    unsigned int hash, t_index = SearchVocab(term_vocab, term_string), e_index;
    // if this term isn't found in the vocabulary, stop and return -1
    if (t_index == -1) return -1;

    // Add the term-entity pairing(s) to the map
    m->map[m->map_size].term = &term_vocab->vocab[t_index];
    m->map[m->map_size].num_entities = num_entities;
    m->map[m->map_size].entities = malloc(num_entities * sizeof(struct indexed_string));
    if (!m->map[m->map_size].entities) {
        error("   >>> Failed to allocate memory for entity storage\n");
        exit(1);
    }

    for (i = 0; i < num_entities; i++) {
        e_index = SearchVocab(entity_vocab, entity_ids[i]);
        if (e_index == -1) {
            e_index = AddWordToVocab(entity_vocab, entity_ids[i]);
        }

        m->map[m->map_size].entities[i].string = entity_vocab->vocab[e_index].word;
        m->map[m->map_size].entities[i].vocab_index = e_index;
    }
    m->map_size++;

    // Reallocate memory if needed
    if (m->map_size + 100 >= m->map_max_size) {
        m->map_max_size += 1000;
        m->map = (struct term_entities *)realloc(m->map, m->map_max_size * sizeof(struct term_entities));
    }
    for (i = m->map_size; i < m->map_max_size; i++) {
        m->map[i].entities = NULL;
    }
    
    // Mark the index in the term hash
    hash = GetWordHash(term_string);
    while (m->term_hash[hash] != -1) hash = (hash + 1) % term_hash_size;
    m->term_hash[hash] = m->map_size - 1;

    // clean up memory from entity parsing
    for (i = 0; i < num_entities; i++) {
        if (entity_ids[i] != NULL) {
            free(entity_ids[i]);
            entity_ids[i] = NULL;
        }
    }
    if (entity_ids != NULL) {
        free(entity_ids);
        entity_ids = NULL;
    }

    // And return the new term's index
    return m->map_size - 1;
}

struct entity_map *ReadEntityMap(char *mapfile, struct vocabulary *term_vocab,
        struct vocabulary *entity_vocab, char *map_sep) {
    long long i = 0;
    char term[MAX_STRING], entities[1000*MAX_STRING];
    FILE *fin = fopen(mapfile, "rb");
    if (fin == NULL) {
        error("Map file not found\n");
        exit(1);
    }
    struct entity_map *m = CreateEntityMap();
    while (1) {
        ReadWord(term, fin, MAX_STRING);
        ReadWord(entities, fin, 1000*MAX_STRING);
        if (feof(fin)) break;
        verbose("  --> Read %s\t%s\n", term, entities);
        AddTermToMap(m, term, entities, term_vocab, entity_vocab, map_sep);
        i++;
        verbose("   >>> Read %lld mappings...\n", i);
    }
    fclose(fin);
    info("Map size: %ld                 \n", m->map_size);
    return m;
}

/**
 * Returns position of a term in the term->entity map; if term not found, returns -1
 */
int SearchMap(struct entity_map *m, struct vocab_word *term) {
    unsigned int hash = GetWordHash(term->word);
    while (1) {
        if ((m->term_hash)[hash] == -1) return -1;
        if (m->map[m->term_hash[hash]].term == term) return m->term_hash[hash];
        hash = (hash + 1) % term_hash_size;
    }
    return -1;
}

/**
 * Identifies the maximum number of entities mapped to from any term in the entity_map
 */
int MaxNumEntities(struct entity_map *m) {
    int max_num = 0, this_num;
    for (long i = 0; i < m->map_size; i++) {
        this_num = m->map[i].num_entities;
        if (this_num > max_num) max_num = this_num;
    }
    return max_num;
}
