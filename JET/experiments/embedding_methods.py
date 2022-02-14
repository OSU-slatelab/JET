'''
Constants for different baseline methods
'''

import sys
import codecs
import numpy as np
from collections import OrderedDict
from hedgepig_logger import log
from embedding_wrapper import EmbeddingWrapper
from drgriffis.common import lutil
import pyemblib

NONE = -1
ENTITY = NONE
WORD = 0
TERM = 1

def readFilterSet(f):
    filter_set = set()
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            filter_set.add(line.strip().lower())
    return filter_set

def readTermStringMap(f, term_ids=None):
    if term_ids:
        id_set = set(term_ids)
        term_filter = lambda t: t in id_set
    else:
        term_filter = lambda t: True

    mapped = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (term_id, term_str) = [s.strip() for s in line.split('\t')]
            if term_filter(term_id):
                mapped[term_id] = term_str
    return mapped

def readTermEntityMap(f, entity_ids=None, term_ids=None, map_sep=','):
    if entity_ids:
        ent_id_set = set(entity_ids)
        ent_filter = lambda e: e in ent_id_set
    else:
        ent_filter = lambda e: True

    if term_ids:
        term_id_set = set(term_ids)
        term_filter = lambda t: t in term_id_set
    else:
        term_filter = lambda t: True

    mapped = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (term_id, ent_list) = [s.strip() for s in line.split('\t')]
            if term_filter(term_id):
                ent_ids = [s.strip() for s in ent_list.split(map_sep)]
                for ent_id in ent_ids:
                    if ent_filter(ent_id):
                        if not term_id in mapped: mapped[term_id] = set()
                        mapped[term_id].add(ent_id)
    return mapped

def name(bl):
    if bl == NONE:
        return 'Entity embeddings'
    elif bl == WORD:
        return 'Word embeddings'
    elif bl == TERM:
        return 'Term embeddings'

def reflect(name):
    glob = globals()
    res = glob.get(name, None)
    if res is None or not type(res) is int:
        return None
    else:
        return res

def CLIOptions(parser):
    parser.add_option('--representation-method', dest='repr_method',
            help='method for representing candidates with specified embeddings file(s)',
            default='NONE')
    parser.add_option('--entities', dest='ent_embf',
            help='path to embedding file for entities')
    parser.add_option('--terms', dest='term_embf',
            help='path to embedding file for terms')
    parser.add_option('--words', dest='word_embf',
            help='path to embedding file for words')
    parser.add_option('--term-map', dest='termmapf',
            help='map from terms to entities they represent')
    parser.add_option('--term-map-sep', dest='term_map_sep',
            help='character separating the list of entities a term corresponds to (default: %default)',
            default=',')
    parser.add_option('--string-map', dest='strmapf',
            help='map from term IDs to their string forms')
    parser.add_option('--entity-filter', dest='ent_filterf',
            help='path to file with filtered list of entity IDs to read')
    parser.add_option('--term-filter', dest='term_filterf',
            help='path to file with filtered list of term IDs to read')
    parser.add_option('--word-filter', dest='word_filterf',
            help='path to file with filtered list of words to read')
    parser.add_option('--keep-word-case', dest='keep_word_case',
            action='store_true', default=False,
            help='keeps word embeddings case-sensitive (all words are lowercased by default)')

def logCLIOptions(options):
    return OrderedDict([
        ('Entity file', options.ent_embf),
        ('Term file', options.term_embf),
        ('Word file', options.word_embf),
        ('Entity filter', options.ent_filterf),
        ('Term filter', options.term_filterf),
        ('Word filter', options.word_filterf),
        ('Keeping word case', options.keep_word_case),
        ('Term-entity map', options.termmapf),
        ('Term-entity map separator', options.term_map_sep),
        ('Term-string map', options.strmapf),
    ])

def validateCLIOptions(options):
    repr_methd_str = options.repr_method
    options.repr_method = reflect(options.repr_method)
    if options.repr_method is None:
        raise Exception('Unknown representation method %s' % repr_methd_str)

    if options.repr_method == TERM and options.strmapf is None:
        raise Exception('--string-map is required when using --representation-method=%s' % repr_methd_str)

    if options.repr_method == ENTITY and options.ent_embf is None:
        raise Exception('--entities is required when using --representation-method=%s' % repr_methd_str)

    if options.repr_method == TERM and options.term_embf is None:
        raise Exception('--terms is required when using --representation-method=%s' % repr_methd_str)

    if options.repr_method == WORD and options.word_embf is None:
        raise Exception('--words is required when using --representation-method=%s' % repr_methd_str)

    if options.ent_filterf and not options.ent_embf:
        raise Exception('--entities is required when using --entity-filter')
    if options.term_filterf and not options.term_embf:
        raise Exception('--terms is required when using --term-filter')
    if options.word_filterf and not options.word_embf:
        raise Exception('--words is required when using --word-filter')

def getEmbeddings(options, log=log, separator=' '):
    word_embs, term_embs, ent_embs = None, None, None
    word_ids, term_ids, ent_ids = None, None, None

    # load in embeddings
    if options.ent_embf:
        if options.ent_filterf: filter_set = readFilterSet(options.ent_filterf)
        else: filter_set = None
        t_sub = log.startTimer('Reading entity embeddings from %s...' % options.ent_embf, newline=False)
        ent_embs = pyemblib.read(options.ent_embf, separator=separator, replace_errors=True, filter_to=filter_set, lower_keys=True)
        log.stopTimer(t_sub, message='Read %d embeddings ({0:.2f}s)' % len(ent_embs))
        ent_ids = ent_embs.keys()
    if options.term_embf:
        if options.term_filterf: filter_set = readFilterSet(options.term_filterf)
        else: filter_set = None
        t_sub = log.startTimer('Reading term embeddings from %s...' % options.term_embf, newline=False)
        term_embs = pyemblib.read(options.term_embf, separator=separator, replace_errors=True, filter_to=filter_set, lower_keys=True)
        log.stopTimer(t_sub, message='Read %d embeddings ({0:.2f}s)' % len(term_embs))
        term_ids = term_embs.keys()
    if options.word_embf:
        if options.word_filterf: filter_set = readFilterSet(options.word_filterf)
        else: filter_set = None
        t_sub = log.startTimer('Reading word embeddings from %s...' % options.word_embf, newline=False)
        word_embs = pyemblib.read(options.word_embf, separator=separator, replace_errors=True, filter_to=filter_set, lower_keys=(not options.keep_word_case))
        log.stopTimer(t_sub, message='Read %d embeddings ({0:.2f}s)' % len(word_embs))
        word_ids = word_embs.keys()

    # load in term/string maps
    if options.termmapf:
        t_sub = log.startTimer('Reading term-entity mappings from %s (separated by "%s")...' % (options.termmapf, options.term_map_sep), newline=False)
        term_entity_map = readTermEntityMap(options.termmapf, entity_ids=ent_ids, term_ids=term_ids, map_sep=options.term_map_sep)
        log.stopTimer(t_sub, message='Read mappings for %d terms ({0:.2f}s)' % len(term_entity_map))
    if options.strmapf:
        t_sub = log.startTimer('Reading term-string mappings from %s...' % options.strmapf, newline=False)
        term_string_map = readTermStringMap(options.strmapf, term_ids=term_ids)
        log.stopTimer(t_sub, message='Read mappings for %d terms ({0:.2f}s)' % len(term_string_map))

    # perform actual approximations
    if options.repr_method == ENTITY:
        emb_wrapper = EmbeddingWrapper(options.repr_method, ent_embs, indexed=True)

    elif options.repr_method == TERM:
        # rekey term embeddings
        new_term_embs = {}
        for (term_id, term_emb) in term_embs.items():
            term_str = term_string_map.get(term_id, None)
            if term_str:
                new_term_embs[term_str] = term_emb
        emb_wrapper = EmbeddingWrapper(options.repr_method, new_term_embs, backoff_embeds=word_embs, indexed=True)

    elif options.repr_method == WORD:
        if options.term_embf:
            raise Exception("Honestly, I don't think this setting is used.")
        else:
            emb_wrapper = EmbeddingWrapper(options.repr_method, word_embs, indexed=True)

    else:
        raise Exception("Huh? %s" % options.repr_method)

    return emb_wrapper

def approximate(child_set, child_embeds):
    members = []
    for child in child_set:
        child_embed = child_embeds.get(child, None)
        if not child_embed is None:
            members.append(child_embed)
    if len(members) > 0:
        return np.mean(members, axis=0)
    else:
        return None
