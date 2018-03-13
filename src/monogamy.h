#ifndef _monogamy_h
#define _monogamy_h

struct term_monogamy {
    float *by_word;
    float monogamy_weight;
    int num_tokens;
};

struct term_monogamy_map {
    struct term_monogamy *monogamies;
    long map_size;
};

struct term_monogamy_map *CreateMonogamyMap(long num_terms);
void DestroyMonogamyMap(struct term_monogamy_map **map);
void CalculateMonogamy(struct vocabulary *term_vocab, struct vocabulary *word_vocab,
        struct term_string_map *strmap, long long unigram_corpus_size,
        struct term_monogamy_map *map);

#endif
