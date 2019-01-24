'''
Script for converting a binary word2vec file to text format
'''

from . import read
from . import word2vec
from .common import *

formats = {
    'word2vec-binary' : 'Binary word2vec format',
    'word2vec-text'   : 'Text word2vec format',
}

if __name__ == '__main__':
    
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog EMBEDF OUTPUTF',
                description='Reads binary word2vec-format embeddings in BINF and writes them as text to TXTF')
        parser.add_option('--from', dest='from_format',
            type='choice', choices=list(formats.keys()),
            help='current format of EMBEDF')
        parser.add_option('--to', dest='to_format',
            type='choice', choices=list(formats.keys()),
            help='desired output format of OUTPUTF')
        (options, args) = parser.parse_args()
        if len(args) != 2:
            parser.print_help()
            exit()
        return args[0], args[1], options.from_format, options.to_format
    srcf, destf, from_format, to_format = _cli()

    print('== Embedding format conversion ==')
    print('  Input %s format file: %s' % (from_format, srcf))
    print('  Output %s format file: %s' % (to_format, destf))

    print('\nReading %s input...' % from_format)
    if from_format == 'word2vec-binary':
        embeddings = read(srcf, format=Format.Word2Vec, mode=Mode.Binary)
    elif from_format == 'word2vec-text':
        embeddings = read(srcf, format=Format.Word2Vec, mode=Mode.Text)
    print('Writing %s output...' % to_format)
    if to_format == 'word2vec-binary':
        word2vec.write(embeddings, destf, mode=Mode.Binary, verbose=True)
    elif to_format == 'word2vec-text':
        word2vec.write(embeddings, destf, mode=Mode.Text, verbose=True)
