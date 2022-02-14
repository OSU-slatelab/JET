#ifndef _mem_h
#define _mem_h

void *MallocOrDie(size_t mem_size, char *msg);
void FreeAndNull(void **ptr);

#endif
