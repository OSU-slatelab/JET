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
from preprocessing.ngram import NGramMapper, NGramMapPrinter
from dependencies.drgriffis.common import preprocessing as pre
from dependencies.drgriffis.common import pickleio
from dependencies.drgriffis.common.logging import log

def readStringsFile(stringf, remove_stopwords=False, use_collapsed_string=False):
    if remove_stopwords:
        stopwords = set(nltk.corpus.stopwords.words('english'))
    else:
        stopwords = set()

    hook = codecs.open(stringf, 'r', 'utf-8')

    # initialize tag->[cuis] and ngram maps
    strcuis = {}
    mapper = NGramMapper()

    hook.readline() # ignore header line
    log.track(message='  >> Processed {0:,} lines', writeInterval=1000)
    for line in hook:
        (cui, string) = line.split('\t')
        string, cui = pre.tokenize(string), cui.strip().lower()
        # if the string is in the set of stopwords to ignore, ignore it
        if ' '.join(string) in stopwords:
            continue

        # add string to ngram map
        ID = mapper.add(string, use_collapsed_string=use_collapsed_string)
        # add CUI to this tag ID
        cuis = strcuis.get(ID, [])
        cuis.append(cui)
        strcuis[ID] = list(set(cuis)) # remove duplicates

        log.tick()
    log.writeln()

    hook.close()

    return mapper.ngrams, strcuis

def writeTagCUIMap(strcuis, outf, sep=','):
    hook = codecs.open(outf, 'w', 'utf-8')
    for (tagID, cuis) in strcuis.items():
        hook.write('%s\t%s\n' % (tagID, sep.join(cuis)))
    hook.close()

def writeTagStrMap(ngrams, outf):
    hook = codecs.open(outf, 'w', 'utf-8')
    _writeTagStrMapHelper(ngrams, hook)
    hook.close()

def _writeTagStrMapHelper(ngrams, hook):
    for mapped in ngrams.values():
        # if we found a complete tag, write it
        if type(mapped) == tuple:
            (string, ID, children) = mapped
            hook.write('%s\t%s\n' % (ID, string))
            _writeTagStrMapHelper(children, hook)
        else:
            _writeTagStrMapHelper(mapped, hook)

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog STRINGFILE [PICKLEDIR] [OPTIONS]')
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
        (options, args) = parser.parse_args()
        if len(args) != 1 and len(args) != 2:
            parser.print_help()
            exit()
        stringf = args[0]
        if len(args) < 2: pickledir = os.path.dirname(stringf)
        else: pickledir = args[1]
        return stringf, pickledir, options.sep, options.remove_stopwords, options.verbose, options.use_collapsed_string, options.logfile

    def _logstart(args):
        (stringf, pickledir, sep, remove_stopwords, _, use_collapsed_string) = args
        log.writeln('Reading STR -> CUI file.')
        log.writeln('  Strings in: %s' % stringf)
        log.writeln('  Storing maps in: %s' % pickledir)
        log.writeln('  Map concepts separated by: \'%s\'' % sep)
        log.writeln('  Removing stopword terms: %s' % log.yesno(remove_stopwords))
        if use_collapsed_string:
            log.writeln('  [DEBUG] Using collapsed strings: Yes')
        log.writeln()

    sys.setrecursionlimit(1800)

    (stringf, pickledir, sep, remove_stopwords, verbose, use_collapsed_string, logfile) = args = _cli()
    log.start(logfile=logfile, message=_logstart, args=args[:-1])

    t_sub = log.startTimer('Reading strings...')
    ngrams, strcuis = readStringsFile(stringf, remove_stopwords=remove_stopwords, use_collapsed_string=use_collapsed_string)
    log.stopTimer(t_sub)

    if verbose:
        log.writeln('\nRead map:')
        NGramMapPrinter.prn(ngrams)

        log.writeln('\nTag-CUI mapping:')
        for tag in strcuis.keys():
            log.writeln('  %s -> %s' % (tag, strcuis[tag]))

    picklebase = os.path.join(pickledir, os.path.basename(stringf))

    tagmapf = '%s.strmap.txt' % picklebase
    t_sub = log.startTimer('Writing tag string map to %s...' % tagmapf)
    writeTagStrMap(ngrams, tagmapf)
    log.stopTimer(t_sub)

    picklef, strf = '%s.tagmap.pkl.gz' % picklebase, '%s.tagmap.txt' % picklebase
    t_sub = log.startTimer('Storing tag-CUI map: Text->%s   Pickled->%s...' % (picklef, strf))
    writeTagCUIMap(strcuis, strf, sep=sep)
    pickleio.write(strcuis, picklef)
    log.stopTimer(t_sub)

    picklef = '%s.ngram.pkl.gz' % picklebase
    t_sub = log.startTimer('\nPickling ngrams to %s...' % picklef)
    pickleio.write(ngrams, picklef)
    log.stopTimer(t_sub)

    if verbose:
        log.writeln('  Unpickling...')
        ngrams2 = pickleio.read(picklef)

        log.writeln('\nUnpickled map:')
        NGramMapPrinter.prn(ngrams2)
