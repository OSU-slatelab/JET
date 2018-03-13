#include <stdlib.h>
#include "logging.h"
#include "mem.h"

void *MallocOrDie(size_t mem_size, char *msg) {
    void *ptr = malloc(mem_size);
    if (!ptr) {
        error("   >>> Failed to allocate memory for %s; Aborting\n", msg);
        exit(1);
    }
    return ptr;
}

void FreeAndNull(void **ptr) {
    if (*ptr != NULL) {
        free(*ptr);
        *ptr = NULL;
    }
}
