'''
'''

import numpy as np
from . import mention_file
from .datasets import nlm_wsd
from .datasets import aida
from . import datasets
from drgriffis.common import log, util
import nltk
from nltk.corpus import stopwords

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog [options] OUTFILE',
                description='Generates common-format file giving mentions (with contextual information) for full dataset')
        parser.add_option('--word-vocab', dest='word_vocabf',
                help='list of in-vocabulary words (if not supplied, all words assumed in vocab)')
        parser.add_option('--entity-vocab', dest='entity_vocabf',
                help='list of known entities to restrict to (if not supplied, all entities assumed in vocab)')
        parser.add_option('--allow-oov', dest='allowoov',
                help='allow OOV words in the context and mentions',
                action='store_true', default=False)
        parser.add_option('--allow-oov-mention-words', dest='allow_oov_mention_words',
                help='allow OOV words in the mentions only (overridden by --allow-oov)',
                action='store_true', default=False)
        parser.add_option('-w', '--window', dest='window',
                help='number of words on each side to count as context window (default: %default)',
                type='int', default=5)
        parser.add_option('--filter-stopwords', dest='filter_stopwords',
                action='store_true', default=False)
        parser.add_option('--dataset', dest='dataset',
                type='choice', choices=[datasets.NLM_WSD, datasets.AIDA])
        (options, args) = parser.parse_args()
        if len(args) != 1:
            parser.print_help()
            exit()
        (outf,) = args
        return outf, options.word_vocabf, options.entity_vocabf, options.allowoov, options.allow_oov_mention_words, options.window, options.dataset, options.filter_stopwords

    outf, w_vocabf, e_vocabf, allow_oov, allow_oov_mention_words, window_size, dataset, filter_stopwords = _cli()

    if w_vocabf:
        log.writeln('Getting word filtering vocabulary from %s...' % w_vocabf)
        word_vocab = set([w.lower() for w in util.readList(w_vocabf, encoding='utf-8')])
        word_filter = lambda w: allow_oov or (w.lower() in word_vocab)
    else:
        word_filter = lambda w: True

    if filter_stopwords:
        stops = set(stopwords.words('english'))
        old_word_filter = word_filter
        word_filter = lambda w: old_word_filter(w) and (not w in stops)

    if e_vocabf:
        log.writeln('Getting concept filtering vocabulary from %s...' % e_vocabf)
        entity_vocab = set([c.lower() for c in util.readList(e_vocabf, encoding='utf-8')])
        entity_filter = lambda c: c.lower() in entity_vocab
    else:
        entity_filter = lambda c: True

    if dataset == datasets.NLM_WSD:
        t_sub = log.startTimer('Generating NLM WSD features.')
        dataset = nlm_wsd.NLM_WSD()
        mentions = nlm_wsd.getAllMentions(dataset, window_size, word_filter, entity_filter)
        log.stopTimer(t_sub, 'Extracted %d samples.' % len(mentions))
    elif dataset == datasets.AIDA:
        t_sub = log.startTimer('Generating AIDA features.')
        dataset = aida.AIDA()
        mentions = aida.getAllMentions(dataset, window_size, word_filter, entity_filter)
        log.stopTimer(t_sub, 'Extracted %d samples.' % len(mentions))

    t_sub = log.startTimer('Writing samples to %s...' % outf, newline=False)
    mention_file.write(mentions, outf)
    log.stopTimer(t_sub, message='Done ({0:.2f}s).')
