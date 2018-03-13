#ifndef _pre_init_h
#define _pre_init_h

void InitTermsFromWords(real *term_embeds, struct vocabulary *term_vocab,
    real *word_embeddings, struct vocabulary *word_vocab,
    struct term_string_map *strmap, struct term_monogamy_map *monomap,
    long long embedding_size);

#endif
