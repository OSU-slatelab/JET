#ifndef _io_h
#define _io_h

struct indexed_string {
    char *string;
    int vocab_index;
};

struct term_annotation {
    long long start_offset;       // number of tokens after start of previous annotation
    int num_tokens;               // number of tokens in this annotation
    int tokens_so_far;            // counter of how many tokens we've processed
    struct indexed_string *term;  // the annotated term
    int *contexts;                // container for observed context words
    int *member_words;            // container for word indices in the term
    int buffer_index;             // tracker for where in the I/O buffer this annotation is
};

void ReadWord(char *word, FILE *fin, int max_strlen);
void ReadCorpusWord(char *word, FILE *hook, int max_strlen);
void ReadAnnotation(struct term_annotation *annot, FILE *fin);
long long GetFileSize(char *fname);
bool FileExists(char *fname);
char *EOSToken();

#endif
