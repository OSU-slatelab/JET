import codecs
import array
import numpy as np
from .common import *

class GloveMode:
    IgnoreContexts = 0
    SumContexts = 1
    GetContexts = 2

def read(fname, vocab=None, mode=GloveMode.SumContexts, include_bias=False, size_only=True):
    if not vocab:
        raise Exception("vocab must be specified for GloVe embeddings")

    words, vectors = [], []

    # get the embedding vocabulary
    if type(vocab) is str:
        h = codecs.open(vocab, 'r', 'utf-8')
        for line in h:
            words.append(line.strip().split()[0])
        h.close()
    else:
        words = vocab.copy()

    inf = open(fname, 'rb')

    # set up for parsing the stored numbers
    real_size = 8  # default double precision
    file_size = getFileSize(inf)
    dim = int((float(file_size) / (real_size * len(words))) / 2)

    if size_only:
        return (len(words), dim)

    # extract the stored vectors
    for i in range(len(words)):
        vector = array.array('d', inf.read(dim*2*real_size))
        if mode == GloveMode.IgnoreContexts:
            vector = vector[:dim]
            if not include_bias: vector = vector[:-1]
        elif mode == GloveMode.SumContexts:
            vector = array.array('d',
                np.array(vector[:dim]) + np.array(vector[dim:])
            )
            if not include_bias: vector = vector[:-1]
        elif mode == GloveMode.GetContexts:
            vector = (vector[:dim], vector[dim:])
            if not include_bias: vector = (vector[0][:-1], vector[1][:-1])
        vectors.append(vector)

    inf.close()

    return (words, vectors)
