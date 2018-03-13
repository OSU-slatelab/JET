#ifndef _logging_h
#define _logging_h

#define DEBUG 3
#define VERBOSE 2
#define INFO 1
#define ERROR 0

extern int LOGLEVEL;

void set_log_level(int level);
void error(const char *format, ...);
void verbose(const char *format, ...);
void info(const char *format, ...);
void debug(const char *format, ...);

#endif
