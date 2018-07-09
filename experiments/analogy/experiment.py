import os
import sys
import numpy as np
from .BMASS import settings
from .analogy_model import Mode
from .task import analogyTask
from . import datasets
import embedding_methods as em
import configlogger
from drgriffis.common import log

def evaluate(emb_wrapper, analogy_file, dataset, setting, analogy_type, analogy_method, log=log,
        predictions_file=None, report_top_k=5):
    t_main = log.startTimer()

    results = analogyTask(analogy_file, dataset, setting, analogy_type, emb_wrapper, log=log,
        predictions_file=predictions_file, predictions_file_mode='w', report_top_k=report_top_k)

    log.stopTimer(t_main, message='Program complete in {0:.2f}s.')

    return results


if __name__ == '__main__':

    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog ANALOGY_FILE',
                description='Run the analogy task on analogies in ANALOGY_FILE')
        em.CLIOptions(parser)
        parser.add_option('--dataset', dest='dataset',
                help='analogy dataset to run on',
                type='choice', choices=[datasets.BMASS, datasets.Google])
        parser.add_option('--setting', dest='setting',
                help='BMASS variant',
                type='choice', choices=['All-Info', 'Multi-Answer', 'Single-Answer'])
        parser.add_option('--type', dest='anlg_type',
                type='choice', choices=['concept', 'string'],
                default='concept')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='logfile')
        parser.add_option('--predictions-file', dest='predictions_file',
                help='file to write predictions for individual analogies to')
        parser.add_option('--predictions-top-k', dest='report_top_k',
                help='number of predictions to log in the predictions file (default: %default)',
                type='int', default=5)
        parser.add_option('--analogy-method', dest='analogy_method',
                help='method to use for analogy completion',
                type='int', default=Mode.ThreeCosAdd)
        parser.add_option('--tab-separated', dest='tab_sep',
                action='store_true', default=False,
                help='use if embeddings file uses \\t to separate key from value')
        (options, args) = parser.parse_args()
        if len(args) != 1:
            parser.print_help()
            exit()

        (analogy_file,) = args

        if options.setting == 'Single-Answer': options.setting = settings.SINGLE_ANSWER
        elif options.setting == 'Multi-Answer': options.setting = settings.MULTI_ANSWER
        elif options.setting == 'All-Info': options.setting = settings.ALL_INFO

        # only one valid setting for Google dataset
        if options.dataset == datasets.Google:
            options.setting = settings.SINGLE_ANSWER

        em.validateCLIOptions(options)

        #return (analogy_file, results_dir, options)
        return (analogy_file, options)
    
    (analogy_file, options) = _cli()
    log.start(logfile=options.logfile, stdout_also=True)

    configlogger.writeConfig(log, settings=[
        ('Dataset', options.dataset),
        ('Dataset file', analogy_file),
        ('Analogy setting', settings.name(options.setting)),
        ('Analogy type', options.anlg_type),
        ('Method', Mode.name(options.analogy_method)),
        ('Embedding settings', em.logCLIOptions(options)),
    ], title='Analogy completion task')

    separator = '\t' if options.tab_sep else ' '
    emb_wrapper = em.getEmbeddings(options, log=log, separator=separator)

    results = evaluate(emb_wrapper, analogy_file, options.dataset, options.setting,
        options.anlg_type, options.analogy_method, log=log, predictions_file=options.predictions_file,
        report_top_k=options.report_top_k)
