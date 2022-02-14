#ifndef _vocab_learner_h
#define _vocab_learner_h

struct vocabulary *GetWordVocabulary(char *plaintextf, char *vocabf);
struct vocabulary *GetTermVocabulary(char *annotationsf, char *vocabf);

struct vocabulary *LearnWordVocabulary(char *plaintextf, char *vocabf, bool overwrite);
struct vocabulary *LearnTermVocabulary(char *annotationsf, char *vocabf, bool overwrite);

#endif
