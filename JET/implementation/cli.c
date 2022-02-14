#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/**
 * Gets the index of a command-line option with argument in argv;
 * If option not found, returns -1
 */
int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

/**
 * Gets the index of a command-line flag (no argument) in argv;
 * If flag not found, returns -1
 */
int FlagPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        return a;
    }
    return -1;
}
