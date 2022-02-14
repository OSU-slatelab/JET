'''
Entity-level analysis of the effect of polysemy on errorneous decisions.

For each dataset, extracts the ranking of predicted pairs, the corpus
polysemy of each entity in the pair, and the ranking error.
Writes these to a file.

Finally, performs linear regression over the entity polysemies w/r/t
ranking error, and reports r^2.
'''

import codecs
import csv
from sklearn import linear_model
from hedgepig_logger import log

def readPolysemy(f):
    polysemy = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        for record in reader:
            polysemy[record['Entity'].lower()] = float(record['CorpusPolysemy'])
    return polysemy

def readResults(f):
    results = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        in_predicted_block, current_dataset = False, None
        for line in stream:
            # ignore blank lines
            if len(line.strip()) == 0: continue

            # check for a section header
            if line[:2] == '==':
                (dataset, section_title) = [s.strip() for s in line.strip(' \n=').split('::')]
                if section_title == 'Predicted order':
                    in_predicted_block = True
                    current_dataset = dataset
                    assert not current_dataset in results
                    results[current_dataset] = []
                else:
                    in_predicted_block = False

            elif in_predicted_block:
                left, right = line.split('-->')
                
                # parse out entity/string pair
                e_1, middle, s_2 = [s.strip(' "') for s in left.split(':')]
                s_1_end = middle.index('"')
                s_1 = middle[:s_1_end]
                e_2 = middle[s_1_end+1:].strip()

                # grab ranking error
                (_, error) = [s.strip(' \n]') for s in right.split(':')]

                results[current_dataset].append((
                    e_1, s_1, e_2, s_2, int(error)
                ))

    return results

def addPolysemy(results, polysemy):
    for (dataset, dataset_res) in results.items():
        extended_res = []
        for (e_1, s_1, e_2, s_2, error) in dataset_res:
            if not e_1 in polysemy:
                print("[WARNING] Entity '%s' not found in polysemy, skipping" % e_1)
                continue
            if not e_2 in polysemy:
                print("[WARNING] Entity '%s' not found in polysemy, skipping" % e_2)
                continue
            p_1 = polysemy[e_1.lower()]
            p_2 = polysemy[e_2.lower()]
            extended_res.append((e_1, s_1, p_1, e_2, s_2, p_2, error))
        results[dataset] = extended_res

def writePolyErrors(dataset_res, f):
    with codecs.open(f, 'w', 'utf-8') as stream:
        stream.write('%s\n' % ('\t'.join(['Entity1', 'String1', 'Entity2', 'String2', 'Polysemy1', 'Polysemy2', 'RankingError'])))
        dataset_res.sort(key=lambda k:k[-1], reverse=True)
        for (e_1, s_1, p_1, e_2, s_2, p_2, error) in dataset_res:
            stream.write('%s\n' % ('\t'.join([
                e_1, s_1,
                e_2, s_2,
                str(p_1), str(p_2),
                str(error)
            ])))

def runRegression(dataset_res):
    observations, labels = [], []
    for (e_1, s_1, p_1, e_2, s_2, p_2, error) in dataset_res:
        observations.append((p_1, p_2))
        labels.append(error)
    reg = linear_model.LinearRegression()
    reg.fit(observations, labels)

    return (
        reg.coef_,
        reg.intercept_,
        reg.score(observations, labels)
    )

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog RESULTSF POLYSEMYF OUTF')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()
        if len(args) != 3:
            parser.print_help()
            exit()
        if not options.logfile: options.logfile = '%s.analysis.log' % args[2]
        return args, options.logfile
    (resultsf, polysemyf, outf), logfile = _cli()
    log.start(logfile=logfile, stdout_also=True)

    log.writeConfig([
        ('Results file', resultsf),
        ('Polysemy file', polysemyf),
        ('Output files', outf)
    ], 'Running sim/rel error analysis')

    results = readResults(resultsf)
    polysemy = readPolysemy(polysemyf)
    addPolysemy(results, polysemy)

    for (dataset, dataset_res) in results.items():
        log.writeln('\nDataset: %s' % dataset)
        outfile = '%s.%s.tsv' % (outf, dataset)
        writePolyErrors(dataset_res, outfile)
        log.writeln('  Wrote errors w/ polysemy to: %s' % outfile)

        (coefs, intercept, r_sq) = runRegression(dataset_res)
        log.writeln('\n  Linear regression output:')
        log.writeln('    Coefficients -- %s' % str(coefs))
        log.writeln('    Intercept -- %s' % str(intercept))
        log.writeln('    R-Squared -- %f' % r_sq)

    log.stop()
