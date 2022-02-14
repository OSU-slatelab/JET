GCC=gcc
CFLAGS=-I. -lm -Ofast -pthread -march=native -Wall -funroll-loops -Wno-unused-result
SHELL=/bin/bash

JET/implementation/bin/JET: JET/implementation/JET.c JET/implementation/thread_config.c JET/implementation/vocab_learner.c JET/implementation/vocab.c JET/implementation/io.c JET/implementation/parallel_reader.c JET/implementation/logging.c JET/implementation/cli.c JET/implementation/entities.c JET/implementation/model_io.c JET/implementation/model.c JET/implementation/context_manager.c JET/implementation/term_strings.c JET/implementation/mem.c
	@set -e; \
	if [ ! -d JET/implementation/bin ]; then mkdir JET/implementation/bin; fi; \
	${GCC} -g \
		JET/implementation/vocab.c JET/implementation/entities.c JET/implementation/io.c JET/implementation/logging.c JET/implementation/mem.c JET/implementation/cli.c JET/implementation/parallel_reader.c JET/implementation/term_strings.c JET/implementation/mt19937ar.c \
		JET/implementation/thread_config.c JET/implementation/vocab_learner.c JET/implementation/model_io.c JET/implementation/model.c JET/implementation/context_manager.c \
		JET/implementation/JET.c \
		-I src \
		-o JET/implementation/bin/JET \
		${CFLAGS}

demo_annotation:
	@set -e; \
	$${PY} -m tagcorpus \
		--threads=3 \
		data/corpus \
		data/strings \
		data/corpus.annotations \
		--max-lines=100
