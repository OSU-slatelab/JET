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
from ..dependencies.drgriffis.common import pickleio
from .ngram import NGramMapper, NGramMapPrinter
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
        (entity_ID, term) = line.split('\t')[:2]
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

def compileTerminology(
    input_filepath,
    output_basepath,
    tokenizer=tokenization.Spacy,
    spacy_model='en_core_web_sm',
    remove_stopwords=False,
    use_collapsed_string=False,
    multi_concept_separator=',',
    verbose=False,
    recursion_limit=10000
):
    '''
    Prepare a terminology for progressive token-by-token scanning of an input
    corpus.

    Given an input tab-separated terminology file and a base path for output
    files, this method creates the following:
    (1) <output_base>.ngram_term_map.pkl.gz :: pickle stored NGramMapper object
        for mapping input text to terms in the terminology, token by token.
        Each term is mapped to a unique ID.
    (2) <output_base>.term_to_entity_map.pkl.gz :: pickle stored dictionary
        mapping from terms in the terminology to the set of entities each term
        can represent.
    (3) <output_base>.term_to_string_map.txt :: tab-separated file mapping term
        IDs in the NGramMapper to full string forms.
    (4) <output_base>.term_to_entity_map.txt :: tab-separated file mapping term
        IDs in the NGramMapper to the set of entity IDs they are mapped to in
        the source terminology.

    Positional arguments:
    input_filepath -- path to the input tab-separated terminology file
        of form "<term>\\t<entity>"
    output_basepath -- base path for generating output files (see descriptions
        of files that will be generated above)

    Keyword arguments:
    tokenizer -- string indicating the tokenizer to use for processing
        multi-token terms (refers to ./tokenization.py)
    spacy_model -- string indicating the spaCy model to be loaded if spaCy
        tokenization is used (this argument is ignored if another tokenizer
        is used)
    remove_stopwords -- boolean indicating whether stopwords should be removed
        from terms (using the NLTK stopwords list)
    use_collapsed_string -- boolean indicating whether multi-token strings
        should be treated as a single, unique token (this prevents a longer
        term from including a shorter term)
    multi_concept_separator -- string used to separate multiple concepts that
        a single term can map to (in <output_base>.term_to_entity_map.txt)
    verbose -- boolean indicating whether the compiled NGramMapper should be
        printed to stdout
    recursion_limit -- used to call sys.setrecursionlimit()

    Return values:
    ngrams -- compiled NGramMapper object
    entities_by_term -- dictionary mapping terms to the entities they
        can represent
    '''
    sys.setrecursionlimit(recursion_limit)

    t_sub = log.startTimer('Initializing tokenizer...')
    tokenizer = tokenization.Tokenizer.build(
        tokenizer,
        spacy_model=spacy_model
    )
    log.stopTimer(t_sub, message='Tokenizer ready in {0:.2f}s.\n')

    t_sub = log.startTimer('Reading terminology file...')
    ngrams, entities_by_term = readTerminology(
        input_filepath,
        tokenizer,
        remove_stopwords=remove_stopwords,
        use_collapsed_string=use_collapsed_string
    )
    log.stopTimer(t_sub, message='Completed in {0:.2f}s.\n')

    if verbose:
        log.writeln('\nRead map:')
        NGramMapPrinter.prn(ngrams)

        log.writeln('\nTerm ID-Entity mapping:')
        for term_ID in entities_by_term.keys():
            log.writeln('  %s -> %s' % (term_ID, entities_by_term[term_ID]))

    term_to_string_map_f = '%s.term_to_string_map.txt' % output_basepath
    t_sub = log.startTimer('Writing term ID-string map to %s...' % term_to_string_map_f)
    writeTermStringMap(ngrams, term_to_string_map_f)
    log.stopTimer(t_sub)

    picklef = '%s.term_to_entity_map.pkl.gz' % output_basepath
    strf =  '%s.term_to_entity_map.txt' % output_basepath
    t_sub = log.startTimer('Storing term ID-entity map: Text->%s   Pickled->%s...' % (picklef, strf))
    writeTermEntityMap(entities_by_term, strf, sep=multi_concept_separator)
    pickleio.write(entities_by_term, picklef)
    log.stopTimer(t_sub)

    picklef = '%s.ngram_term_map.pkl.gz' % output_basepath
    t_sub = log.startTimer('\nPickling ngrams to %s...' % picklef)
    pickleio.write(ngrams, picklef)
    log.stopTimer(t_sub)

    if verbose:
        log.writeln('  Unpickling...')
        ngrams2 = pickleio.read(picklef)

        log.writeln('\nUnpickled map:')
        NGramMapPrinter.prn(ngrams2)

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

    options = args = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Terminology file', options.input_f),
        ('Storing pickled maps to', options.output_dir),
        ('Map concepts separated by', options.sep),
        ('Removing stopword terms', options.remove_stopwords),
        ('Tokenization settings', tokenization.CLI.logOptions(options)),
    ], 'JET -- STR -> CUI file preprocessing')

    picklebase = os.path.join(
        options.output_dir,
        os.path.splitext(os.path.basename(options.input_f))[0]
    )

    compileTerminology(
        input_filepath=options.input_f,
        output_basepath=picklebase,
        tokenizer=options.tokenizer_type,
        spacy_model=options.tokenizer_spacy_model,
        remove_stopwords=options.remove_stopwords,
        use_collapsed_string=options.use_collapsed_string,
        verbose=options.verbose,
        recursion_limit=1000
    )

    log.stop()
