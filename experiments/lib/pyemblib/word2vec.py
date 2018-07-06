import codecs
import array
import sys
import numpy as np
from .common import *

def read(fname, mode=Mode.Binary, size_only=False, first_n=None, separator=' ', replace_errors=False, filter_to=None):
    '''Returns array of words and word embedding matrix
    '''
    if mode == Mode.Text: output = _readTxt(fname, size_only=size_only, first_n=first_n, filter_to=filter_to)
    elif mode == Mode.Binary: output = _readBin(fname, size_only=size_only, first_n=first_n, separator=separator, replace_errors=replace_errors, filter_to=filter_to)
    return output

def _readTxt(fname, size_only=False, first_n=None, filter_to=None):
    '''Returns array of words and word embedding matrix
    '''
    words, vectors = [], []
    hook = codecs.open(fname, 'r', 'utf-8')

    if filter_to:
        filter_set = set([f.lower() for f in filter_to])
        key_filter = lambda k: k.lower() in filter_set
    else:
        key_filter = lambda k: True

    # get summary info about vectors file
    (numWords, dim) = (int(s.strip()) for s in hook.readline().split())
    if size_only:
        return (numWords, dim)

    for line in hook:
        if len(line.strip()) > 0:
            chunks = line.split()
            try:
                word, vector = chunks[0].strip(), np.array([float(n) for n in chunks[1:]])
            except Exception as e:
                print("<<< CHUNKING ERROR >>>")
                print(line)
                raise e
            if len(vector) == dim - 1:
                sys.stderr.write("[WARNING] Read vector without a key, skipping\n")
            elif len(vector) != dim:
                raise ValueError("Read %d-length vector, expected %d" % (len(vector), dim))
            else:
                if key_filter(word):
                    words.append(word)
                    vectors.append(vector)

                if (not first_n is None) and len(words) == first_n:
                    break
    hook.close()

    if not first_n is None:
        assert len(words) == first_n
    elif not filter_to:
        if len(words) != numWords:
            sys.stderr.write("[WARNING] Expected %d words, read %d\n" % (numWords, len(words)))

    return (words, vectors)

def _getFileSize(inf):
    curIx = inf.tell()
    inf.seek(0, 2)  # jump to end of file
    file_size = inf.tell()
    inf.seek(curIx)
    return file_size

def _readBin(fname, size_only=False, first_n=None, separator=' ', replace_errors=False, filter_to=None):
    import sys
    words, vectors = [], []

    if filter_to:
        filter_set = set([f.lower() for f in filter_to])
        key_filter = lambda k: k.lower() in filter_set
    else:
        key_filter = lambda k: True

    inf = open(fname, 'rb')

    # get summary info about vectors file
    summary = inf.readline().decode('utf-8')
    summary_chunks = [int(s.strip()) for s in summary.split(' ')]
    (numWords, dim) = summary_chunks[:2]
    if len(summary_chunks) > 2: float_size = 8
    else: float_size = 4

    if size_only:
        return (numWords, dim)

    # make best guess about byte size of floats in file
    #float_size = 4

    bsep = separator.encode('utf-8')

    chunksize = 10*float_size*1024
    curIx, nextChunk = inf.tell(), inf.read(chunksize)
    #while len(nextChunk) > 0 and len(words) < numWords:
    while len(nextChunk) > 0:
        inf.seek(curIx)

        splitix = nextChunk.index(bsep)
        #print('splitIx: %d   nextChunk: %s' % (splitix, nextChunk[:splitix]))
        bts = inf.read(splitix)
        if replace_errors:
            word = bts.decode('utf-8', errors='replace')
        else:
            word = bts.decode('utf-8')
        #word = inf.read(splitix).decode('utf-8', errors='replace')
        #print('word: %s' % word)
        inf.seek(1,1) # skip the space
        vector = np.array(array.array('f', inf.read(dim*float_size)))
        #print(vector)
        inf.seek(1,1) # skip the newline

        if key_filter(word):
            words.append(word)
            vectors.append(vector)
        curIx, nextChunk = inf.tell(), inf.read(chunksize)
        #print('curIx: %d' % curIx)
        #input()
        #sys.stdout.write('  >> Read %d words\r' % len(words))

        if (not first_n is None) and len(words) == first_n:
            break

    inf.close()

    # verify that we read properly
    if not first_n is None:
        assert len(words) == first_n
    elif not filter_to:
        assert len(words) == numWords
    return (words, vectors)

def write(embeds, fname, mode=Mode.Binary, verbose=False):
    '''Writes a dictionary of embeddings { term : embed }
    to a file, in the format specified.
    '''
    if mode == Mode.Binary:
        outf = open(fname, 'wb')
        write_str = lambda s: outf.write(s.encode('utf-8'))
    elif mode == Mode.Text:
        outf = codecs.open(fname, 'w', 'utf-8')
        write_str = lambda s: outf.write(s)

    wordmap = embeds

    # write summary info
    keys = list(wordmap.keys())
    vdim = 0 if len(keys) == 0 else len(wordmap.get(keys[0]))
    write_str('%d %d\n' % (len(keys), vdim))

    if verbose:
        sys.stdout.write(' >>> Writing %d-d embeddings for %d words\n' % (vdim, len(keys)))
        sys.stdout.flush()

    # write vectors
    ctr = {'count':0}
    def tick(ctr):
        ctr['count'] += 1
        if ctr['count'] % 1000 == 0:
            sys.stdout.write('\r >>> Written %d/%d words' % (ctr['count'], len(keys)))
            if ctr['count'] % 5000 == 0:
                sys.stdout.flush()

    if mode == Mode.Binary:
        test_emb = wordmap.get(keys[0])
        if 'astype' in dir(test_emb): write_op = lambda e, o: e.astype('f').tofile(o)
        else: write_op = lambda e, o: np.float32(e).tofile(o)

        for word in keys:
            write_str(word)
            embedding = wordmap.get(word)
            outf.write(b' ')
            write_op(embedding, outf)
            write_str('\n')
            tick(ctr)
    elif mode == Mode.Text:
        for word in keys:
            write_str(word)
            embedding = wordmap.get(word)
            for f in embedding: write_str(' %.8f' % f)
            write_str('\n')
            tick(ctr)
    outf.close()

    if verbose:
        sys.stdout.write('\r >>> Written %d/%d words\n' % (ctr['count'], len(keys)))
        sys.stdout.flush()
