#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "logging.h"

int LOGLEVEL;

void set_log_level(int level) {
    LOGLEVEL = level;
}

void verbose(const char *format, ...) {
    if (LOGLEVEL >= VERBOSE) {
        va_list args;
        va_start(args, format);
        vfprintf(stdout, format, args);
        va_end(args);
        fflush(stdout);
    }
}

void info(const char *format, ...) {
    if (LOGLEVEL >= INFO) {
        va_list args;
        va_start(args, format);
        vfprintf(stdout, format, args);
        va_end(args);
        fflush(stdout);
    }
}

void error(const char *format, ...) {
    if (LOGLEVEL >= ERROR) {
        va_list args;
        va_start(args, format);
        vfprintf(stdout, format, args);
        va_end(args);
        fflush(stdout);
    }
}

void debug(const char *format, ...) {
    if (LOGLEVEL >= DEBUG) {
        va_list args;
        va_start(args, format);
        vfprintf(stdout, format, args);
        va_end(args);
        fflush(stdout);
    }
}
