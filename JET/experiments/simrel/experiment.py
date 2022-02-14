'''
Run word similarity/relatedness evaluations on embeddings and save results.
'''

import numpy as np
import glob
import os
import codecs
from collections import OrderedDict
from hedgepig_logger import log
import embedding_methods as em
from .datasets import *
from .simrel_logging import logPredictions
from scipy.stats import spearmanr
from drgriffis.science.metrics import CosineSimilarity
from drgriffis.common import util

class Mode:
    UMNSRS = 'UMNSRS'
    WikiSRS = 'WikiSRS'

def prepare(dataset):
    prepped = []
    for pairing in dataset:
        prepared_item = [pairing[i].lower() for i in range(len(pairing)-1)]
        prepared_item.append(pairing[-1])
        prepped.append(prepared_item)
    return prepped

def readSkips(f):
    skips = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            # skip comments
            if line[0] == '#': continue
            else:
                (ds, ix) = [s.strip() for s in line.split(',')]
                if not ds in skips:
                    skips[ds] = set()
                skips[ds].add(int(ix))
    return skips

def evaluateOn(dataset, emb_wrapper, sim_metric, log_predictions=False, skips_f=None):
    # check to see how many dataset items are comparable
    comparable, full_comparable = [], []
    prepared = prepare(dataset.data)

    if skips_f:
        skips = readSkips(skips_f)
        skips = skips.get(dataset.name, set())
    else:
        skips = set()

    for i in range(len(prepared)):
        if i in skips: continue
        eval_datum, full_datum = prepared[i], dataset.full_data[i]
        (item1, item2, _) = eval_datum
        if emb_wrapper.knows(item1) and emb_wrapper.knows(item2):
            comparable.append(eval_datum)
            full_comparable.append(full_datum)

    gold, pred = [], []
    for (item1, item2, gold_metric) in comparable:
        gold.append(gold_metric)
        pred.append(sim_metric(emb_wrapper[item1], emb_wrapper[item2]))

    if log_predictions:
        logPredictions(full_comparable, pred, gold, dataset.name, log=log)

    (rho, _) = spearmanr(gold, pred)
    return rho, len(comparable), len(dataset.data)

def twoModelEvaluate(dataset, ent_emb_wrapper, str_emb_wrapper, sim_metric, log_predictions=False, use_cross=False, cross_only=False, use_mean=False, skips_f=None):
    log.writeln('\n\n  Using cross: %s\n  Using cross only: %s\n  Using mean: %s\n' % (str(use_cross), str(cross_only), str(use_mean)))

    # check to see how many dataset items are comparable
    comparable, full_comparable = [], []
    prepared = prepare(dataset.full_data)

    if skips_f:
        skips = readSkips(skips_f)
        skips = skips.get(dataset.name, set())
    else:
        skips = set()

    for i in range(len(prepared)):
        if i in skips: continue
        full_datum = prepared[i]
        (e_1, str_1, e_2, str_2, _) = full_datum
        if ent_emb_wrapper.knows(e_1) and ent_emb_wrapper.knows(e_2) \
                and str_emb_wrapper.knows(str_1) and str_emb_wrapper.knows(str_2):
            comparable.append(full_datum)
            full_comparable.append(dataset.full_data[i])
        else:
            log.writeln('SKIPPING %d' % i)

    gold, pred = [], []
    for (e_1, str_1, e_2, str_2, gold_metric) in comparable:
        gold.append(gold_metric)
        scores = [
            sim_metric(ent_emb_wrapper[e_1], ent_emb_wrapper[e_2]),
            sim_metric(str_emb_wrapper[str_1], str_emb_wrapper[str_2])
        ]
        if use_cross:
            if cross_only:
                scores = []
            scores.append(sim_metric(ent_emb_wrapper[e_1], str_emb_wrapper[str_2]))
            scores.append(sim_metric(str_emb_wrapper[str_1], ent_emb_wrapper[e_2]))
        if use_mean:
            pred.append(np.mean(scores))
        else:
            pred.append(np.sum(scores))

    if log_predictions:
        logPredictions(full_comparable, pred, gold, dataset.name, log=log)

    (rho, _) = spearmanr(gold, pred)
    return rho, len(comparable), len(dataset.data)


if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog EMBEDDINGS',
                description='Runs term similarity/relatedness evaluation on a specified EMBEDDINGS file.')
        em.CLIOptions(parser)
        parser.add_option('--mode', dest='mode',
                help='evaluate on UMNSRS or WikiSRS datasets (default: %default)',
                type='choice', choices=[Mode.UMNSRS, Mode.WikiSRS], default=Mode.UMNSRS)
        parser.add_option('--tab-separated', dest='tab_sep',
                action='store_true', default=False,
                help='use if emeddings file uses \\t to separate key from value')
        parser.add_option('--combo', dest='use_combo',
                help='flag to use both entity and string embeddings together'
                     ' (overrides --representation-method)',
                action='store_true', default=False)
        parser.add_option('--cross', dest='use_cross',
                help='flag to use cross entity/string similarity in addition',
                action='store_true', default=False)
        parser.add_option('--cross-only', dest='cross_only',
                help='flag to use cross entity/string similarity only',
                action='store_true', default=False)
        parser.add_option('--mean-scores', dest='use_mean',
                action='store_true', default=False,
                help='flag to use mean of multiple scores instead of sum')
        parser.add_option('--skip-indices', dest='skips_f',
                help='file listing indices to skip for each dataset (filtered experiments)')
        parser.add_option('-l', '--logfile', dest='logfile')
        (options, args) = parser.parse_args()

        if options.use_combo and (options.ent_embf is None or options.word_embf is None):
            raise ValueError('--entities and --words both required when using --combo')
        elif not options.use_combo:
            em.validateCLIOptions(options)

        if len(args) != 0:
            parser.print_help()
            exit()

        return options

    options = _cli()
    log.start(logfile=options.logfile, stdout_also=True)

    log.writeConfig(settings=[
        ('Dataset', options.mode),
        ('Using skip indices', ('None' if not options.skips_f else options.skips_f)),
        ('Embedding settings', em.logCLIOptions(options)),
        ('Scoring settings', OrderedDict([
            ('Combination of entity and string', options.use_combo),
            ('Cross comparison of entity/string', options.use_cross),
            ('Cross comparison only', options.cross_only),
            ('Using mean of scores instead of sum', options.use_mean)
        ])),
    ], title='Similarity/Relatedness experiment')

    if not options.use_combo:
        log.writeln('\nMode: %s   Method: %s\n' % (options.mode, em.name(options.repr_method)))
        separator = '\t' if options.tab_sep else ' '
        emb_wrapper = em.getEmbeddings(options, log=log, separator=separator)
    else:
        log.writeln('\nMode: %s   Method: COMBO\n' % options.mode)
        ent_embf, word_embf = options.ent_embf, options.word_embf
        separator = '\t' if options.tab_sep else ' '

        options.repr_method = em.ENTITY
        options.word_embf = None
        ent_emb_wrapper = em.getEmbeddings(options, log=log, separator=separator)

        options.repr_method = em.WORD
        options.ent_embf = None
        options.word_embf = word_embf
        str_emb_wrapper = em.getEmbeddings(options, log=log, separator=separator)

    if options.mode == Mode.UMNSRS:
        datasets = (
            UMNSRS_Similarity(options.repr_method),
            UMNSRS_Relatedness(options.repr_method)
        )
    elif options.mode == Mode.WikiSRS:
        datasets = (
            WikiSRS_Similarity(options.repr_method),
            WikiSRS_Relatedness(options.repr_method)
        )

    t_sub = log.startTimer('Running similarity/relatedness evaluation...', newline=False)
    if options.use_combo:
        results = [
            twoModelEvaluate(dataset, ent_emb_wrapper, str_emb_wrapper, CosineSimilarity,
                    log_predictions=True, use_cross=options.use_cross, cross_only=options.cross_only,
                    skips_f=options.skips_f)
                for dataset in datasets
        ]
    else:
        results = [
            evaluateOn(dataset, emb_wrapper, CosineSimilarity, log_predictions=True, skips_f=options.skips_f)
                for dataset in datasets
        ]
    log.stopTimer(t_sub, message='Done in {0:.2f}s')

    log.writeln('\nResults:')
    for i in range(len(datasets)):
        (rho, compared, ttl) = results[i]
        log.writeln('  %s --> %.4f (%d/%d)' % (datasets[i].name, rho, compared, ttl))

    log.stop()
