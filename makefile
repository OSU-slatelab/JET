GCC=gcc
CFLAGS=-I. -lm -Ofast -pthread -march=native -Wall -funroll-loops -Wno-unused-result
SHELL=/bin/bash

bin/JET: src/JET.c src/thread_config.c src/vocab_learner.c src/vocab.c src/io.c src/parallel_reader.c src/logging.c src/cli.c src/entities.c src/model_io.c src/model.c src/context_manager.c src/term_strings.c src/mem.c
	@set -e; \
	if [ ! -d bin ]; then mkdir bin; fi; \
	${GCC} -g \
		src/vocab.c src/entities.c src/io.c src/logging.c src/mem.c src/cli.c src/parallel_reader.c src/term_strings.c src/mt19937ar.c \
		src/thread_config.c src/vocab_learner.c src/model_io.c src/model.c src/context_manager.c \
		src/JET.c \
		-I src \
		-o bin/JET \
		${CFLAGS}

demo_annotation:
	@set -e; \
	$${PY} -m tagcorpus \
		--threads=3 \
		data/corpus \
		data/strings \
		data/corpus.annotations \
		--max-lines=100
