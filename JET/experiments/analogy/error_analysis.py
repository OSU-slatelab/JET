'''
Compare entity-level and word-level predictions on analogy data, to check for
oracle accuracy in case of always being able to pick the correct source.
'''

import codecs
import numpy as np
import scipy.stats

from .BMASS import settings
from .BMASS import parser as BMASS_parser

ORACLE_ENT = 'Entities'
ORACLE_WRD = 'Words'

def readPreferredStrings(f):
    strmap = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (cui, string) = [s.strip() for s in line.split('\t')]
            strmap[cui.lower()] = string
    return strmap

def readOutputsFromLogfile(logf):
    outputs = {}
    cur_relation, cur_outputs = None, []
    in_header = False
    cur_anlg, cur_result, cur_top_k = None, None, []
    with codecs.open(logf, 'r', 'utf-8') as stream:
        for line in stream:
            if line[:5] == '-----':
                if cur_anlg:
                    cur_outputs.append((
                        cur_anlg,
                        cur_result,
                        cur_top_k
                    ))
                    cur_anlg, cur_result, cur_top_k = None, None, []
                if in_header:
                    in_header = False
                else:
                    in_header = True
            elif in_header:
                if cur_relation:
                    outputs[cur_relation] = cur_outputs
                cur_relation = line.strip()
                cur_outputs = []
            else:
                if len(line.strip()) == 0:
                    if cur_anlg:
                        cur_outputs.append((
                            cur_anlg,
                            cur_result,
                            cur_top_k
                        ))
                    cur_anlg, cur_result, cur_top_k = None, None, []
                else:
                    if not cur_anlg:
                        (a,b,_,c,d) = [s.strip() for s in line.split(':')]
                        cur_anlg = (a,b,c,d)
                    elif not cur_result:
                        cur_result = line.split()[1].strip()
                    else:
                        cur_top_k.append(line.strip())

        if cur_anlg:
            cur_outputs.append((
                cur_anlg,
                cur_result,
                cur_top_k
            ))
        if cur_relation:
            outputs[cur_relation] = cur_outputs
    return outputs


def addInfoFromLogfile(logf, analogies, anlg_ref, info_key, preferred_strings):
    '''
    anlg_ref = 0 for CUIs, 1 for STRs
    '''
    outputs = readOutputsFromLogfile(logf)
    for (rel, anlgs) in analogies.items():
        rel_outputs = outputs[rel]

        outputs_ix, gold_ix = 0, 0
        while outputs_ix < len(rel_outputs) and gold_ix < len(anlgs):
            (rel_output_anlg, rel_output_result, rel_output_preds) = rel_outputs[outputs_ix]
            (gold_anlg, gold_info) = anlgs[gold_ix]

            if rel_output_anlg[0].lower() == gold_anlg[0][anlg_ref].lower() and rel_output_anlg[1].lower() == gold_anlg[1][anlg_ref].lower() and rel_output_anlg[2].lower() == gold_anlg[2][anlg_ref].lower():
                # add strings to entity-level predictions
                if anlg_ref == 0:
                    rel_output_preds_clarified = []
                    for p in rel_output_preds:
                        rel_output_preds_clarified.append('%s (%s)' % (p, preferred_strings.get(p, '')))
                    rel_output_preds = rel_output_preds_clarified
                gold_info[info_key] = (True, rel_output_result, rel_output_preds)
                gold_ix += 1
                outputs_ix += 1
            else:
                gold_info[info_key] = (False, None, None)
                gold_ix += 1


def writeAnalysis(analogies, outf, key_order):
    # calculate oracle information
    oracular_spectacular = {}
    mcnemar_shmcnemar = {}
    for (rel, anlgs) in analogies.items():
        # calculate oracle and McNemar's info
        oracle, mcnemars = {}, {}
        for (_, output_info) in anlgs:
            (ent_saw_it, ent_result, _) = output_info.get(ORACLE_ENT, (False, None, None))
            (wrd_saw_it, wrd_result, _) = output_info.get(ORACLE_WRD, (False, None, None))

            if ent_saw_it and wrd_saw_it:
                if ent_result == 'True' and wrd_result == 'True':
                    mcnemars['a'] = mcnemars.get('a', 0) + 1
                elif ent_result == 'True' and wrd_result == 'False':
                    mcnemars['b'] = mcnemars.get('b', 0) + 1
                elif ent_result == 'False' and wrd_result == 'True':
                    mcnemars['c'] = mcnemars.get('c', 0) + 1
                else:
                    mcnemars['d'] = mcnemars.get('d', 0) + 1
            elif ent_saw_it:
                if ent_result == 'True':
                    mcnemars['b'] = mcnemars.get('b', 0) + 1
                else:
                    mcnemars['d'] = mcnemars.get('d', 0) + 1
            elif wrd_saw_it:
                if wrd_result == 'True':
                    mcnemars['c'] = mcnemars.get('c', 0) + 1
                else:
                    mcnemars['d'] = mcnemars.get('d', 0) + 1
            else:
                mcnemars['d'] = mcnemars.get('d', 0) + 1

            if ent_saw_it and wrd_saw_it:
                oracle['both_saw'] = oracle.get('both_saw', 0) + 1
                if ent_result == 'True' and wrd_result == 'True':
                    oracle['both_saw_both_correct'] = oracle.get('both_saw_both_correct', 0) + 1
                elif ent_result == 'True':
                    oracle['both_saw_ent_correct'] = oracle.get('both_saw_ent_correct', 0) + 1
                elif wrd_result == 'True':
                    oracle['both_saw_wrd_correct'] = oracle.get('both_saw_wrd_correct', 0) + 1
            elif ent_saw_it:
                oracle['ent_only_saw'] = oracle.get('ent_only_saw', 0) + 1
                if ent_result == 'True':
                    oracle['ent_only_correct'] = oracle.get('ent_only_correct', 0) + 1
            elif wrd_saw_it:
                oracle['wrd_only_saw'] = oracle.get('wrd_only_saw', 0) + 1
                if wrd_result == 'True':
                    oracle['wrd_only_correct'] = oracle.get('wrd_only_correct', 0) + 1
            else:
                oracle['neither_saw'] = oracle.get('neither_saw', 0) + 1

        oracle['entity_correct'] = oracle.get('both_saw_both_correct',0) + oracle.get('both_saw_ent_correct',0) + oracle.get('ent_only_correct',0)
        oracle['entity_total'] = oracle.get('both_saw',0) + oracle.get('ent_only_saw',0)
        oracle['entity_accuracy'] = float(oracle['entity_correct'])/len(anlgs)

        oracle['word_correct'] = oracle.get('both_saw_both_correct',0) + oracle.get('both_saw_wrd_correct',0) + oracle.get('wrd_only_correct',0)
        oracle['word_total'] = oracle.get('both_saw',0) + oracle.get('wrd_only_saw',0)
        oracle['word_accuracy'] = float(oracle['word_correct'])/len(anlgs)

        oracle['oracle_correct'] = oracle.get('both_saw_both_correct',0) + oracle.get('both_saw_ent_correct',0) + oracle.get('both_saw_wrd_correct',0) + oracle.get('ent_only_correct',0) + oracle.get('wrd_only_correct',0)
        oracle['oracle_total'] = oracle.get('both_saw',0) + oracle.get('ent_only_saw',0) + oracle.get('wrd_only_saw',0)
        oracle['oracle_accuracy'] = float(oracle['oracle_correct'])/len(anlgs)

        oracular_spectacular[rel] = oracle
        mcnemar_shmcnemar[rel] = mcnemars

    with codecs.open(outf, 'w', 'utf-8') as stream:
        # write out oracle summary
        stream.write('## Oracle summary ########################\n')
        stream.write('\nRel,EntityAcc,WordAcc,OracleAcc\n')
        for (rel, oracle) in oracular_spectacular.items():
            stream.write('%s,%f,%f,%f\n' % (
                rel,
                oracle['entity_accuracy'],
                oracle['word_accuracy'],
                oracle['oracle_accuracy'],
            ))

        # write out McNemar's summary
        stream.write('\n## McNemar\'s summary #####################\n')
        for (rel, mcnemars) in mcnemar_shmcnemar.items():
            b, c = mcnemars.get('b', 0), mcnemars.get('c', 0)
            if b + c == 0: chi_2 = 0
            else:
                chi_2 = ((float(b)-c)**2)/(b+c)
            p_value = 1 - scipy.stats.chi2.cdf(chi_2, df=1)
            stream.write('%s -- [%d %d %d %d] %f (%f)\n' % (rel, mcnemars.get('a', 0), mcnemars.get('b', 0), mcnemars.get('c', 0), mcnemars.get('d', 0), chi_2, p_value))

        for (rel, anlgs) in analogies.items():
            stream.write(('\n{0}\n  %s\n{0}\n'.format('-'*80)) % rel)

            # check which settings got it right
            corrects, seen = {}, {}
            for (_, output_info) in anlgs:
                for key in key_order:
                    (key_saw_it, key_result, _) = output_info.get(key, (False, None, None))
                    if key_saw_it:
                        seen[key] = seen.get(key, 0) + 1
                        if key_result == 'True':
                            corrects[key] = corrects.get(key, 0) + 1

            # summary info
            stream.write('Coverage summary:\n')
            for key in key_order:
                stream.write('  %s : %d/%d (%f)\n' % (key, seen.get(key, 0), len(anlgs), float(seen.get(key, 0))/len(anlgs)))
            stream.write('Accuracy summary:\n')
            for key in key_order:
                stream.write('  %s : %d/%d (%f)\n' % (key, corrects.get(key, 0), seen.get(key,0), float(corrects.get(key, 0))/seen.get(key,1)))

            stream.write('\n== Oracle info ==\n')
            stream.write('Oracle seen: %d (%d entity %d word)\n' % (oracle['oracle_total'], oracle.get('ent_only_saw',0)+oracle.get('both_saw',0), oracle.get('wrd_only_saw',0)+oracle.get('both_saw',0)))
            stream.write('Oracle performance: %d/%d (%f)\n' % (oracle['oracle_correct'], oracle['oracle_total'], oracle['oracle_accuracy']))
            stream.write('\nBoth correct: %d\n' % oracle.get('both_saw_both_correct', 0))
            stream.write('Entity only correct: %d\n' % (oracle.get('both_saw_ent_correct', 0) + oracle.get('ent_only_correct', 0)))
            stream.write('Word only correct: %d\n' % (oracle.get('both_saw_wrd_correct', 0) + oracle.get('wrd_only_correct', 0)))

            stream.write('\n== Per-analogy info ==\n')
            for (anlg, output_info) in anlgs:
                stream.write('\n%s\n' % str(anlg))
                for key in key_order:
                    (key_saw_it, key_result, key_preds) = output_info.get(key, (False, None, None))
                    stream.write('  %s :\n' % key)
                    stream.write('    Saw it: %s\n' % str(key_saw_it))
                    if key_saw_it:
                        stream.write('    Correct: %s\n' % str(key_result))
                        stream.write('    Predictions: %s\n' % str(key_preds))


if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog LOGFILE1 MODE1 LABEL1 [LOGFILE2 MODE2 LABEL2 [...]]')
        parser.add_option('--output', dest='outf',
                help='error analysis output file (required)')
        parser.add_option('--strings', dest='stringf',
                help='file mapping entity IDs to preferred string forms')
        parser.add_option('--analogy-file', dest='analogy_file',
                help='path to source analogy file (required)')
        (options, args) = parser.parse_args()
        if len(args) == 0 or len(args) % 3 != 0 or not options.outf or not options.analogy_file:
            parser.print_help()
            exit()

        triples = []
        for i in range(0,len(args),3):
            triples.append((args[i], args[i+1], args[i+2]))
        return triples, options
    (triples, options) = _cli()

    if options.stringf:
        preferred_strings = readPreferredStrings(options.stringf)
    else:
        preferred_strings = {}

    # get both CUI and string versions of each analogy
    analogies = BMASS_parser.read(options.analogy_file, settings.MULTI_ANSWER, strings_only=False, cuis_only=False)
    # and restructure to accommodate prediction information
    for (rel, anlgs) in analogies.items():
        restruct_anlgs = []
        for anlg in anlgs:
            restruct_anlgs.append((anlg, {}))
        analogies[rel] = restruct_anlgs


    for (logf, mode, label) in triples:
        addInfoFromLogfile(
            logf, 
            analogies,
            0 if mode == 'CUI' else 1,
            label,
            preferred_strings
        )

    writeAnalysis(analogies, options.outf, [t[2] for t in triples])
