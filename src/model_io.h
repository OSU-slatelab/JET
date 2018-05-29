#ifndef _model_io_h
#define _model_io_h

void CreateModelDirectory(char *dirpath);
void WriteVectors(char *f, struct vocabulary *v, float *embeds, long long embed_size, int binary);
void LoadVectors(char *fpath, float *embeddings, struct vocabulary *vocab, long long embed_size);
void WriteTermEntityLikelihoods(char *f, struct vocabulary *tv, struct entity_map *em,
    real *term_entity_likelihoods);
void WriteHyperparameters(char *f, struct hyperparameters params);

#endif
