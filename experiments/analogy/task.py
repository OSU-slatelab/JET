'''
Run trained embeddings through generated analogy task.
'''

import codecs
import numpy as np
import tensorflow as tf
from .BMASS import settings
from . import datasets
from .BMASS import parser as BMASS_parser
from .Google import parser as Google_parser
from .analogy_model import AnalogyModel
from drgriffis.common import log

def completeAnalogySet(str_analogies, setting, emb_wrapper, grph, report_top_k=5, log=log):
    analogies, embedded_analogies, kept_str_analogies = [], [], []

    # if using multi_answer, find the maximum number of answers for any analogy in this set
    if setting in [settings.ALL_INFO, settings.MULTI_ANSWER]:
        max_answers = 0
        for analogy in str_analogies:
            max_answers = max(max_answers, len(analogy[3]))
    else:
        max_answers = 1

    # convert analogies to a matrix of indices and a matrix of embeddings
    log.track(message='  >> Preprocessing: {0}/%d' % len(str_analogies), writeInterval=50)
    for analogy in str_analogies:
        valid, analogy_ixes, analogy_embeds = convertAnalogyToMatrices(analogy, setting, emb_wrapper, max_answers)

        if valid:
            analogies.append(analogy_ixes)
            embedded_analogies.append(analogy_embeds)
            kept_str_analogies.append(analogy)
        log.tick()
    log.flushTracker()

    correct, MAP, MRR, total, skipped, predictions = grph.eval(analogies, embedded_analogies, report_top_k=report_top_k, log=log)
    log.flushTracker(len(analogies))

    str_predictions = []
    for i in range(len(predictions)):
        (is_correct, num_candidates, predicted_ixes) = predictions[i]

        if len(predicted_ixes) > 1:
            predicted_strings = [emb_wrapper.indexToTerm(ix) for ix in predicted_ixes]
        elif predicted_ixes[0] == -1:
            predicted_strings = ['>>> SKIPPED <<<']

        str_predictions.append((
            kept_str_analogies[i],
            is_correct,
            num_candidates,
            predicted_strings
        ))
    return (correct, MAP, MRR, total, skipped, str_predictions)



def convertAnalogyToMatrices(analogy, setting, emb_wrapper, max_answers):
    '''Convert an analogy into
    1) A matrix of indices in the term vocabulary
    2) A matrix of term embeddings
    '''
    valid = True
    # analogy is a:b::c:d
    analogy_ixes, analogy_embeds = [], []
    for i in range(4):
        # if using multi_both and looking at `b`, just ignore it
        if setting == settings.ALL_INFO and i == 1:
            analogy_ixes.append(-3)
        # if only using single answers, handle all 4 components the same
        elif setting == settings.SINGLE_ANSWER or i < 3:
            key = analogy[i]
            ix = emb_wrapper.index(key)
            analogy_ixes.append(ix)
        # if using multiple answers, add on the array of valid answers
        elif setting in [settings.ALL_INFO, settings.MULTI_ANSWER] and i == 3:
            found_at_least_one_answer = False
            for j in range(max_answers):
                if j < len(analogy[3]):
                    ix = emb_wrapper.index(analogy[3][j])
                    if ix > -1: found_at_least_one_answer = True
                    analogy_ixes.append(ix)
                else:
                    analogy_ixes.append(-2) # emb_wrapper can return -1; use -2 to signal All Done
            # if none of the valid answers are in the vocabulary, then skip this analogy
            if not found_at_least_one_answer: valid = False

        try:
            # if using multi_both, use the average of the offsets between
            # `a` and all the possible `b`s
            if setting == settings.ALL_INFO and i == 1:
                keys = analogy[i]
                ixes = [emb_wrapper.index(key) for key in keys]
                avg_embed, valid_bs = np.zeros(len(emb_wrapper[0])), 0
                for j in range(len(ixes)):
                    ix, key = ixes[j], keys[j]
                    # if there's a b_k we can't model, skip it
                    try:
                        if ix > -1: avg_embed += emb_wrapper[ix]
                        else:
                            #log.writeln('[ERROR] USING BAD ACCESS 1: %s' % key)
                            avg_embed += emb_wrapper[key]
                        valid_bs += 1
                    except (KeyError, AttributeError):
                        pass
                # as long as we were able to model at least one b_k, keep this analogy
                if valid_bs > 0: analogy_embeds.append(avg_embed / valid_bs)
                else: valid = False
            # only grab the embeddings for a,b,c
            elif i < 3:
                if ix > -1:
                    analogy_embeds.append(emb_wrapper[ix])
                else:
                    #log.writeln('[ERROR] USING BAD ACCESS 2: %s' % key)
                    analogy_embeds.append(emb_wrapper[key])
        except (KeyError, AttributeError) as e:
            # thrown if we can't get an embedding for something in the analogy;
            # skip this analogy
            valid = False
            print('COULD NOT FIND "%s"' % (analogy[i]))
            break
    
    return (
        valid,
        analogy_ixes,
        analogy_embeds
    )




def analogyTask(analogy_file, dataset, setting, analogy_type, emb_wrapper, log=log, report_top_k=5, predictions_file=None, predictions_file_mode='w'):
    if dataset == datasets.BMASS:
        if analogy_type == 'string':
            analogies = BMASS_parser.read(analogy_file, setting, strings_only=True)
        else:
            analogies = BMASS_parser.read(analogy_file, setting, cuis_only=True)
    elif dataset == datasets.Google:
        analogies = Google_parser.read(analogy_file, to_lower=True)

    # if we're saving the predictions, start that file first
    if predictions_file: pred_stream = codecs.open(predictions_file, predictions_file_mode, 'utf-8')

    # build the analogy completion model
    sess = tf.Session()
    grph = AnalogyModel(sess, emb_wrapper.asArray())

    completed, results = 0, {}
    for (relation, rel_analogies) in analogies.items():
        t_file = log.startTimer('  Starting relation: %s (%d/%d)' % (relation, completed+1, len(analogies)))

        rel_results = completeAnalogySet(rel_analogies, setting, emb_wrapper, grph, report_top_k, log=log)
        results[relation] = rel_results

        (correct, MAP, MRR, total, skipped, predictions) = rel_results
        log.stopTimer(t_file, message='  Completed file: %s (%d/%d) [{0:.2f}s]\n    >> Skipped %d/%d' % (
            relation, completed+1, len(analogies), skipped, total
        ))

        if predictions_file:
            pred_stream.write(('{0}\n  %s\n{0}\n'.format('-'*79)) % relation)
            for prediction in predictions:
                ((a,b,c,d), is_correct, num_candidates, top_k) = prediction
                pred_stream.write('\n%s:%s::%s:%s\nCorrect: %s\nPredictions: %d\n%s\n' % (
                    a,b,c,d,
                    str(is_correct),
                    num_candidates,
                    '\n'.join([('    %s' % guess) for guess in top_k])
                ))

        completed += 1

    # tie off the predictions file
    if predictions_file: pred_stream.close()

    return results
