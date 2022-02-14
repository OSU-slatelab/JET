#ifndef _entities_h
#define _entities_h

struct term_entities {
    struct vocab_word *term;
    struct indexed_string *entities;
    int num_entities;
};

struct entity_map {
    struct term_entities *map;
    int *term_hash;
    long long map_max_size;
    long map_size;
};

struct entity_map *ReadEntityMap(char *map_file, struct vocabulary *term_vocab,
        struct vocabulary *concept_vocab, char *map_sep);
int SearchMap(struct entity_map *m, struct vocab_word *term);
int MaxNumEntities(struct entity_map *m);
void DestroyEntityMap(struct entity_map **m);

struct entity_map *CreateEntityMap();
int AddTermToMap(struct entity_map *m, char *term_string, char *entities,
        struct vocabulary *term_vocab, struct vocabulary *entity_vocab,
        char *sep);

#endif
