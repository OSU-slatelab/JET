'''
Script for converting a binary word2vec file to text format
'''

from . import read
from . import word2vec
from .common import *

if __name__ == '__main__':
    
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog BINF TXTF',
                description='Reads binary word2vec-format embeddings in BINF and writes them as text to TXTF')
        (options, args) = parser.parse_args()
        if len(args) != 2:
            parser.print_help()
            exit()
        return args
    binf, txtf = _cli()

    print('== Word2Vec format conversion ==')
    print('  Input binary format file: %s' % binf)
    print('  Output text format file: %s' % txtf)

    print('\nReading binary input...')
    embeddings = read(binf, format=Format.Word2Vec, mode=Mode.Binary)
    print('Writing text format...')
    word2vec.write(embeddings, txtf, mode=Mode.Text, verbose=True)
