'''
Reads a tab-separated file mapping STR -> CUI.

Cleans and tokenizes each string, and builds a hierarchical map through
possible n-grams to unique IDs.

Pickles n-gram map to file for use in corpus tagging.
'''

import os
import sys
import codecs
import nltk.corpus
from hedgepig_logger import log
from dependencies.drgriffis.common import preprocessing as pre
from dependencies.drgriffis.common import pickleio
from preprocessing.ngram import NGramMapper, NGramMapPrinter
from . import tokenization

def readTerminology(terminology_f, tokenizer, remove_stopwords=False, use_collapsed_string=False):
    if remove_stopwords:
        stopwords = set(nltk.corpus.stopwords.words('english'))
    else:
        stopwords = set()

    hook = codecs.open(terminology_f, 'r', 'utf-8')

    # initialize tag->[cuis] and ngram maps
    entities_by_term = {}
    mapper = NGramMapper()

    hook.readline() # ignore header line
    log.track(message='  >> Processed {0:,} lines', writeInterval=1000)
    for line in hook:
        (entity_ID, term) = line.split('\t')
        term = tokenizer.tokenize(term)
        entity_ID = entity_ID.strip()
        # if the string is in the set of stopwords to ignore, ignore it
        if ' '.join(term) in stopwords:
            continue

        # add string to ngram map
        term_ID = mapper.add(term, use_collapsed_string=use_collapsed_string)
        # add CUI to this tag ID
        entities = entities_by_term.get(term_ID, [])
        entities.append(entity_ID)
        entities_by_term[term_ID] = list(set(entities)) # remove duplicates

        log.tick()
    log.flushTracker()

    hook.close()

    return mapper.ngrams, entities_by_term

def writeTermEntityMap(entities_by_term, outf, sep=','):
    hook = codecs.open(outf, 'w', 'utf-8')
    for (term_ID, entities) in entities_by_term.items():
        hook.write('%s\t%s\n' % (term_ID, sep.join(entities)))
    hook.close()

def writeTermStringMap(ngrams, outf):
    hook = codecs.open(outf, 'w', 'utf-8')
    _writeTermStringMapHelper(ngrams, hook)
    hook.close()

def _writeTermStringMapHelper(ngrams, hook):
    for mapped in ngrams.values():
        # if we found a complete tag, write it
        if type(mapped) == tuple:
            (string, term_ID, children) = mapped
            hook.write('%s\t%s\n' % (term_ID, string))
            _writeTermStringMapHelper(children, hook)
        else:
            _writeTermStringMapHelper(mapped, hook)

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-i', '--input', dest='input_f',
            help='(required) input terminology file')
        parser.add_option('-o', '--output', dest='output_dir',
            help='output directory for compiled terminology files (if not'
                 ' provided, defaults to containing directory of --input file)')
        parser.add_option('-v', '--verbose', dest='verbose',
                help='print NGramMaps read from files',
                action='store_true', default=False)
        parser.add_option('-l', '--logfile', dest='logfile',
                help=str.format('name of file to write log contents to (empty for stdout)'),
                default=None)
        parser.add_option('--map-separator', dest='sep',
                help='separator for concepts in tag-concept map file (default \'%default\')',
                default=',')
        parser.add_option('--ignore-stopword-terms', dest='remove_stopwords',
                action='store_true', default=False,
                help='ignore any terms that consist of a single stopword')
        parser.add_option('--string-identifiers', dest='use_collapsed_string',
                action='store_true', default=False,
                help='[DEBUG OPTION] Use the concatenated tokens of a term as its identifier')
        tokenization.CLI.addOptions(parser)
        (options, args) = parser.parse_args()
        if not options.input_f:
            parser.print_help()
            parser.error('Must provide --input')
        if not options.output_dir:
            options.output_dir = os.path.dirname(options.input_f)
        return options

    sys.setrecursionlimit(1800)

    options = args = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Terminology file', options.input_f),
        ('Storing pickled maps to', options.output_dir),
        ('Map concepts separated by', options.sep),
        ('Removing stopword terms', options.remove_stopwords),
        ('Tokenization settings', tokenization.CLI.logOptions(options)),
    ], 'JET -- STR -> CUI file preprocessing')

    t_sub = log.startTimer('Initializing tokenizer...')
    tokenizer = tokenization.CLI.initializeTokenizer(options)
    log.stopTimer(t_sub, message='Tokenizer ready in {0:.2f}s.\n')

    t_sub = log.startTimer('Reading terminology file...')
    ngrams, entities_by_term = readTerminology(
        options.input_f,
        tokenizer,
        remove_stopwords=options.remove_stopwords,
        use_collapsed_string=options.use_collapsed_string
    )
    log.stopTimer(t_sub, message='Completed in {0:.2f}s.\n')

    if options.verbose:
        log.writeln('\nRead map:')
        NGramMapPrinter.prn(ngrams)

        log.writeln('\nTerm ID-Entity mapping:')
        for term_ID in entities_by_term.keys():
            log.writeln('  %s -> %s' % (term_ID, entities_by_term[term_ID]))

    picklebase = os.path.join(
        options.output_dir,
        os.path.splitext(os.path.basename(options.input_f))[0]
    )

    term_to_string_map_f = '%s.term_to_string_map.txt' % picklebase
    t_sub = log.startTimer('Writing term ID-string map to %s...' % term_to_string_map_f)
    writeTermStringMap(ngrams, term_to_string_map_f)
    log.stopTimer(t_sub)

    picklef, strf = '%s.term_to_entity_map.pkl.gz' % picklebase, '%s.term_to_entity_map.txt' % picklebase
    t_sub = log.startTimer('Storing term ID-entity map: Text->%s   Pickled->%s...' % (picklef, strf))
    writeTermEntityMap(entities_by_term, strf, sep=options.sep)
    pickleio.write(entities_by_term, picklef)
    log.stopTimer(t_sub)

    picklef = '%s.ngram_term_map.pkl.gz' % picklebase
    t_sub = log.startTimer('\nPickling ngrams to %s...' % picklef)
    pickleio.write(ngrams, picklef)
    log.stopTimer(t_sub)

    if options.verbose:
        log.writeln('  Unpickling...')
        ngrams2 = pickleio.read(picklef)

        log.writeln('\nUnpickled map:')
        NGramMapPrinter.prn(ngrams2)

    log.stop()
