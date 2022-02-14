#ifndef _thread_config_h
#define _thread_config_h

struct thread_config {
    int thread_id;
    long long tokens;
    long long start_byte_plain;
    long long start_byte_annot;
    long long start_offset_annot;
};

void ThreadConfigureMaster(char *thread_config_file, char *train_file_plain,
        char *train_file_annot, long long *thread_tokens,
        long long *thread_start_bytes_plain, long long *thread_start_bytes_annot,
        long long *thread_start_offsets_annot, int num_threads, bool force);

void AllocateThreadConfigurations(long long **thread_tokens,
        long long **thread_start_bytes_plain, long long **thread_start_bytes_annot,
        long long **thread_start_offsets_annot, int num_threads);
void DestroyThreadConfigurations(long long **thread_tokens,
        long long **thread_start_bytes_plain, long long **thread_start_bytes_annot,
        long long **thread_start_offsets_annot);

#endif
