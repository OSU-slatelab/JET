#ifndef _term_strings_h
#define _term_strings_h

struct term_string_map {
    struct term_string *strings;
    long map_size;
};
struct term_string {
    char **tokens;
    int num_tokens;
};

struct term_string_map *CreateTermStringMap(long num_terms);
void DestroyTermStringMap(struct term_string_map **map);
void ReadTermStringMap(char *mapfile, struct vocabulary *term_vocab, struct term_string_map *map);

#endif
