'''
See Rastogi et al (2015)

Refer to Steiger (1980) to calculate the appropriate
reference statistics for dynamically-sized datasets
'''

import codecs
import re
from hedgepig_logger import log
from scipy.stats import spearmanr
from ..simrel_logging import logPredictions

def readPredictions(f):
    predictions = {}
    cur_ds, cur_predictions, gold_predictions, i, in_gold_order = None, {}, {}, 0, False
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            # header lines
            if line[:3] == '== ':
                line = line.strip('= \n')
                ds, section = [s.strip() for s in line.split('::')]
                if not in_gold_order:
                    if section.lower() == 'gold order':
                        cur_ds = ds
                        cur_predictions, gold_predictions = {}, {}
                        in_gold_order = True
                else:
                    in_gold_order = False
                    predictions[cur_ds] = (cur_predictions, gold_predictions)
            elif in_gold_order and len(line.strip()) > 0:
                first_half, second_half = line.split('-->')
                first_half = re.sub('\s{2,}', ' ', first_half)
                prediction = float(second_half.split('(')[0])
                cur_predictions[first_half] = prediction
                gold_predictions[first_half] = -(i+1)
                i += 1
    return predictions

def correlation(a, b, dsname):
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    shared_keys = a_keys.intersection(b_keys)

    a_obs, b_obs = [], []
    for k in shared_keys:
        a_obs.append(a[k])
        b_obs.append(b[k])

    # convert the shared keys to dataset format to re-use
    # experiment logging code
    dataset = []
    for key in shared_keys:
        (ent1, lbl1_ent2, lbl2) = [s.strip() for s in key.split(':')]
        (_, lbl1, ent2) = lbl1_ent2.split('"')
        lbl2 = lbl2.strip('"')
        dataset.append((ent1, lbl1, ent2, lbl2, None))
    logPredictions(dataset, a_obs, b_obs, dsname, log=log)

    (rho, _) = spearmanr(a_obs, b_obs)
    return (rho*100, len(shared_keys))

def getStatistics(f1, f2):
    preds1 = readPredictions(f1)
    preds2 = readPredictions(f2)

    for ds in preds1.keys():
        log.writeln(('\n\n{0}\n### %s\n{0}\n\n'.format('#'*80)) % ds)
        (lbl_scores_1, gold_1) = preds1[ds]
        (lbl_scores_2, gold_2) = preds2[ds]

        (ab, ab_size) = correlation(lbl_scores_1, lbl_scores_2, '%s -- A vs B' % ds)
        (at, at_size) = correlation(lbl_scores_1, gold_1, '%s -- A vs GOLD' % ds)
        (bt, bt_size) = correlation(lbl_scores_2, gold_2, '%s -- B vs GOLD' % ds)

        log.writeln("\n -- %s Agreement summary --" % ds)
        log.writeln("  |r_bt - r_at| = %f" % abs(at-bt))
        log.writeln("  r_ab = %f (%d)" % (ab, ab_size))
        log.writeln("  r_at = %f (%d)" % (at, at_size))
        log.writeln("  r_bt = %f (%d)" % (bt, bt_size))

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog LOG1 LOG2')
        parser.add_option('-l', '--logfile', dest='logfile')
        (options, args) = parser.parse_args()
        if len(args) != 2:
            parser.print_help()
            exit()
        return args, options.logfile
    (f1, f2), logfile = _cli()
    log.start(logfile=logfile, stdout_also=True)
    log.writeln('A: %s' % f1)
    log.writeln('B: %s' % f2)
    getStatistics(f1, f2)
    log.stop()
