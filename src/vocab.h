#ifndef _vocab_h
#define _vocab_h

#define MAX_STRING 255

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct vocabulary {
   struct vocab_word *vocab;
   int *vocab_hash;
   long long vocab_max_size; //1000
   long vocab_size;
   long long word_count;
};


int ReadCorpusWordIndex(struct vocabulary *v, FILE *fin);
int ReadWordIndex(struct vocabulary *v, FILE *fin);
int SearchVocab(struct vocabulary *v, char *word);
int AddWordToVocab(struct vocabulary *v, char *word);
void SortAndReduceVocab(struct vocabulary *v, int min_count);
struct vocabulary *CreateVocabulary();
void DestroyVocabulary(struct vocabulary **v);
void SaveVocab(struct vocabulary *v, char *vocab_file);
struct vocabulary *ReadVocab(char *vocab_file);
void IncrementVocabFreq(struct vocabulary *v, int ix);
int GetWordHash(char *word);

#endif
