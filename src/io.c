#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "io.h"

const char *EOS_TOKEN = "</s>";

char *EOSToken() {
    return (char *)EOS_TOKEN;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, int max_strlen) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;   // '\r'
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) break;
            else continue; 
        }
        word[a] = ch;
        a++;
        if (a > max_strlen - 1) a--;   // Truncate too long words
    }
    word[a] = 0; // null terminate
}

/**
 * Read next word from file, treating space/tab/EOL as word-terminal.
 * EOL also considered sentence-terminal (produces EOS_TOKEN)
 */
void ReadCorpusWord(char *word, FILE *hook, int max_strlen) {
    int len = 0, ord;
    while (!feof(hook)) {
        ord = fgetc(hook);
        // skip \r
        if (ord == 13) continue;
        // halt on whitespace (if have seen characters; otherwise ignore)
        if ((ord == ' ') || (ord == '\t') || (ord == '\n')) {
            // if we've read non-whitespace characters, stop reading here
            if (len > 0) {
                // put newlines back into the stream, to mark them as sentence terminal at next read
                if (ord == '\n') ungetc(ord, hook);
                break;
            }

            // otherwise, if this is a newline, mark it as a sentence boundary
            if (ord == '\n') {
                strcpy(word, EOS_TOKEN);
                return;
            }
            // ignore other leading whitespace
            else continue;
        }

        word[len] = ord;
        len++;

        // truncate if too long
        if (len >= max_strlen -1) len--;
    }
    word[len] = 0;
}

/**
 * Read the next standoff annotation into a term_annotation
 */
void ReadAnnotation(struct term_annotation *annot, FILE *fin) {
    fscanf(fin, "%lld", &annot->start_offset);
    fscanf(fin, "%d", &annot->num_tokens);
    fscanf(fin, "%s\n", annot->term->string);
    annot->tokens_so_far = 0;
    annot->contexts = NULL;
    annot->member_words = NULL;
    annot->buffer_index = -1;
}

/**
 * Read the byte length of a file
 */
long long GetFileSize(char *fname) {
    long long fsize;
    FILE *fin = fopen(fname, "rb");
    if (fin == NULL) {
        printf("ERROR: file not found! %s\n", fname);
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    fsize = ftell(fin);
    fclose(fin);
    return fsize;
}

/**
 * Check if a specified file exists
 */
bool FileExists(char *f) {
    FILE *file;
    if (f[0] == 0 || (file = fopen(f, "rb")) == NULL) {
        return false;
    } else {
        fclose(file);
    }
    return true;
}
