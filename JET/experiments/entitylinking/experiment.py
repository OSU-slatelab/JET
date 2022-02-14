'''
'''

import codecs
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from drgriffis.common import log
import scipy.stats
import pyemblib
from . import mention_file
from .params import LLParams, EntityComboMethods
from .sabbir_model_c import LinearSabbirLinkerC

class Indexer:
    def __init__(self, vocab):
        self._vocab = vocab
        self._indices = {vocab[i]:i for i in range(len(vocab))}

    def indexOf(self, key):
        return self._indices.get(key, -1)

    def __getitem__(self, index):
        return self._vocab[index]

class McNemars:
    
    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0

    def run(self):
        if self.b + self.c == 0: chi_2 = 0
        else:
            chi_2 = ((float(self.b)-self.c)**2)/(self.b+self.c)
        p_value = 1 - scipy.stats.chi2.cdf(chi_2, df=1)
        return chi_2, p_value
    

def readPreferredStrings(f):
    pref_strings = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (entity_id, string) = [s.strip() for s in line.split('\t')]
            pref_strings[entity_id.lower()] = string
    return pref_strings

def readPolysemy(f):
    polysemy = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        stream.readline()
        for line in stream:
            (entity_id, poly) = [s.strip() for s in line.split('\t')]
            polysemy[entity_id.lower()] = float(poly)
    return polysemy

def runModel(mentions, entity_embeds, ctx_embeds, minibatch_size, preds_file, debug=False,
        secondary_entity_embeds=None, entity_combo_method=None, using_mention=False,
        preds_file_detailed=None, preferred_strings=None, preds_file_polysemy=None,
        polysemy=None):
    entity_vocab, entity_arr = entity_embeds.toarray()
    ctx_vocab, ctx_arr = ctx_embeds.toarray()
    if secondary_entity_embeds:
        secondary_entity_vocab, secondary_entity_arr = secondary_entity_embeds.toarray()
        secondary_entity_arr_2 = []
        for v in secondary_entity_vocab:
            secondary_entity_arr_2.append(np.array(secondary_entity_embeds[v]))
        secondary_entity_arr_2 = np.array(secondary_entity_arr_2)
    else:
        secondary_entity_vocab, secondary_entity_arr = None, None

    ent_ixer = Indexer(entity_vocab)
    ctx_ixer = Indexer(ctx_vocab)
    if secondary_entity_embeds:
        secondary_ent_ixer = Indexer(secondary_entity_vocab)
    else:
        secondary_ent_ixer = None

    max_num_entities = 0
    for m in mentions:
        if len(m.candidates) > max_num_entities:
            max_num_entities = len(m.candidates)

    max_mention_size = 0
    for m in mentions:
        n_tokens = len(m.mention_text.split())
        if n_tokens > max_mention_size:
            max_mention_size = n_tokens

    window_size = 5
    params = LLParams(
        ctx_vocab_size = len(ctx_vocab),
        ctx_dim = ctx_embeds.size,
        entity_vocab_size = len(entity_vocab),
        entity_dim = entity_embeds.size,
        secondary_entity_vocab_size = (0 if not secondary_entity_embeds else len(secondary_entity_vocab)),
        secondary_entity_dim = (0 if not secondary_entity_embeds else secondary_entity_embeds.size),
        window_size = window_size,
        max_num_entities = max_num_entities,
        max_mention_size = max_mention_size,
        entity_combo_method = entity_combo_method,
        using_mention = using_mention
    )

    session = tf.Session()
    lll = LinearSabbirLinkerC(session, np.array(ctx_arr), np.array(entity_arr), params, debug=debug, secondary_entity_embed_arr=np.array(secondary_entity_arr))

    log.track(message='   >>> Processed {0} batches', writeInterval=10)

    if secondary_entity_embeds:
        ent_vs_sec = McNemars()
        ent_vs_joint = McNemars()
        sec_vs_joint = McNemars()
        joint_vs_oracle = McNemars()

    correct, total = 0., 0
    batch_start = 0
    oracle = {}
    while (batch_start < len(mentions)):
        next_batch_mentions = mentions[batch_start:batch_start + minibatch_size]
        next_batch = [
            prepSample(mention, ent_ixer, ctx_ixer, window_size, max_mention_size, max_num_entities, secondary_ent_ixer=secondary_ent_ixer)
                for mention in next_batch_mentions
        ]

        batch_ctx_window_ixes  = [next_batch[i][0] for i in range(len(next_batch))]
        batch_ctx_window_masks = [next_batch[i][1] for i in range(len(next_batch))]
        batch_mention_ixes     = [next_batch[i][2] for i in range(len(next_batch))]
        batch_mention_masks    = [next_batch[i][3] for i in range(len(next_batch))]
        batch_entity_ixes      = [next_batch[i][4] for i in range(len(next_batch))]
        batch_entity_masks     = [next_batch[i][5] for i in range(len(next_batch))]
        if secondary_entity_embeds:
            batch_secondary_entity_ixes = [next_batch[i][6] for i in range(len(next_batch))]
        else:
            batch_secondary_entity_ixes = None

        results = lll.getPredictions(batch_ctx_window_ixes, batch_ctx_window_masks, batch_entity_ixes,
            batch_entity_masks, batch_secondary_entity_ixes=batch_secondary_entity_ixes,
            batch_mention_ixes=batch_mention_ixes, batch_mention_masks=batch_mention_masks, oracle=True)
        if secondary_entity_embeds:
            (preds, probs, ent_preds, secondary_ent_preds) = results
        else:
            (preds, probs, ent_preds) = results
        for i in range(len(next_batch)):
            (_, _, _, _, ent_ixes, _, _, correct_candidate, mention) = next_batch[i]
            # base accuracy eval
            predicted_ix = ent_ixes[preds[i]]
            if predicted_ix == correct_candidate:
                correct += 1
            total += 1

            # oracle eval
            joint_correct, entity_correct, secondary_correct, oracle_correct = False, False, False, False
            if ent_ixes[ent_preds[i]] == correct_candidate:
                entity_correct = True
                oracle['entity_correct'] = oracle.get('entity_correct', 0) + 1
            if secondary_entity_embeds and ent_ixes[preds[i]] == correct_candidate:
                joint_correct = True
                oracle['joint_correct'] = oracle.get('joint_correct', 0) + 1
            if secondary_entity_embeds and ent_ixes[secondary_ent_preds[i]] == correct_candidate:
                secondary_correct = True
                oracle['secondary_correct'] = oracle.get('secondary_correct', 0) + 1
            if entity_correct or secondary_correct:
                oracle_correct = True
                oracle['oracle_correct'] = oracle.get('oracle_correct', 0) + 1

            # significance tracking
            if secondary_entity_embeds:
                # entity vs secondary
                if entity_correct and secondary_correct:
                    ent_vs_sec.a += 1
                elif entity_correct and (not secondary_correct):
                    ent_vs_sec.b += 1
                elif (not entity_correct) and secondary_correct:
                    ent_vs_sec.c += 1
                else:
                    ent_vs_sec.d += 1
                # entity vs joint
                if entity_correct and joint_correct:
                    ent_vs_joint.a += 1
                elif entity_correct and (not joint_correct):
                    ent_vs_joint.b += 1
                elif (not entity_correct) and joint_correct:
                    ent_vs_joint.c += 1
                else:
                    ent_vs_joint.d += 1
                # secondary vs joint
                if secondary_correct and joint_correct:
                    sec_vs_joint.a += 1
                elif secondary_correct and (not joint_correct):
                    sec_vs_joint.b += 1
                elif (not secondary_correct) and joint_correct:
                    sec_vs_joint.c += 1
                else:
                    sec_vs_joint.d += 1
                # joint vs oracle
                if joint_correct and oracle_correct:
                    joint_vs_oracle.a += 1
                elif joint_correct and (not oracle_correct):
                    joint_vs_oracle.b += 1
                elif (not joint_correct) and oracle_correct:
                    joint_vs_oracle.c += 1
                else:
                    joint_vs_oracle.d += 1

            # predictions + scores
            if preds_file:
                preds_file.write('Probs: [ %s ]  Pred: %d -> %d  Gold: %d\n' % (' '.join([str(p) for p in probs[i]]), preds[i], ent_ixes[preds[i]], correct_candidate))

            # predictions + corpus polysemy of correct entity
            if preds_file_polysemy:
                try:
                    line = '%d\t%f\n' % (
                        (1 if predicted_ix == correct_candidate else 0),
                        polysemy[ent_ixer[predicted_ix]]
                    )
                    preds_file_polysemy.write(line)
                except KeyError:
                    pass

            # predictions, in detail
            if preds_file_detailed:
                keys = ['all']
                if secondary_entity_embeds:
                    pred_ixes = [
                        ('Pred (Joint)', ent_ixes[preds[i]]),
                        ('Pred (Ent)', ent_ixes[ent_preds[i]]),
                        ('Pred (Defn)', ent_ixes[secondary_ent_preds[i]])
                    ]
                    if entity_correct and secondary_correct:
                        comp_stream_key = 'both_correct'
                    elif entity_correct and (not secondary_correct):
                        comp_stream_key = 'entity_only_correct'
                    elif (not entity_correct) and secondary_correct:
                        comp_stream_key = 'secondary_only_correct'
                    else:
                        comp_stream_key = 'both_wrong'
                    keys.append(comp_stream_key)
                    #if entity_correct and secondary_correct and joint_correct:
                    #    joint_stream_key = None
                    #if entity_correct and secondary_correct and (not joint_correct):
                    #    joint_stream_key = 'ent_sec_no-joint'
                    #elif entity_correct and joint_correct and (not secondary_correct):
                    #    joint_stream_key = 'ent_and_joint'
                    #elif (not entity_correct) and joint_correct and secondary_correct:
                    #    joint_stream_key = 'sec_and_joint'
                    #elif joint_correct and (not entity_correct) and (not secondary_correct):
                    #    joint_stream_key = 'joint_only'
                    #elif entity_correct and (not joint_correct) and (not secondary_correct):
                    #    joint_stream_key = 'ent_no-joint'
                    #elif (not entity_correct) and (not joint_correct) and secondary_correct:
                    #    joint_stream_key = 'sec_no-joint'
                    #elif (not entity_correct) and (not joint_correct) and (not secondary_correct):
                    #    joint_stream_key = None
                    #keys.append(joint_stream_key)
                    if (not entity_correct) and joint_correct:
                        keys.append('ent_joint_help')
                    elif entity_correct and (not joint_correct):
                        keys.append('ent_joint_hurt')
                    if (not secondary_correct) and joint_correct:
                        keys.append('sec_joint_help')
                    if secondary_correct and (not joint_correct):
                        keys.append('sec_joint_hurt')
                else:
                    pred_ixes = [
                        ('Pred', predicted_ix)
                    ]
                    if entity_correct:
                        stream_key = 'entity_correct'
                    else:
                        stream_key = 'entity_wrong'
                    keys.append(stream_key)
                for k in keys:
                    _writeDetailedOutcome(preds_file_detailed[k], mention, probs, batch_entity_ixes,
                        batch_entity_masks, ent_ixer, preferred_strings, correct_candidate, pred_ixes, i)

        batch_start += minibatch_size
        log.tick()
    log.flushTracker()

    for (msg, mcn) in [
                ('Entity vs Defn', ent_vs_sec),
                ('Entity vs Joint', ent_vs_joint),
                ('Defn vs Joint', sec_vs_joint),
                ('Joint vs Oracle', joint_vs_oracle)
            ]:
        chi2, pval = mcn.run()
        log.writeln('\n%s\n'
                    '    | a = %5d | b = %5d |\n'
                    '    | c = %5d | d = %5d |\n'
                    '  Chi^2 = %f  P-value = %f\n' % (
            msg,
            mcn.a, mcn.b, mcn.c, mcn.d,
            chi2, pval
        ))

    return correct, total, oracle

def _writeDetailedOutcome(stream, mention, probs, batch_entity_ixes, batch_entity_masks, ent_ixer,
        preferred_strings, correct_candidate, pred_ixes, i):
    stream.write(
        '\n-----------------------------------------------------\n'
        'Mention\n'
        '  Left context: %s\n'
        '  Mention text: %s\n'
        '  Right context: %s\n'
        '  Candidates: [ %s ]\n'
        '  Correct answer: %s\n'
        '\nPredictions\n' % (
            mention.left_context,
            mention.mention_text,
            mention.right_context,
            ', '.join(mention.candidates),
            mention.CUI
        )
    )
    for j in range(len(probs[i])):
        if batch_entity_masks[i][j] == 1:
            entity_str = ent_ixer[batch_entity_ixes[i][j]]
            if preferred_strings:
                entity_str = '%s (%s)' % (entity_str, preferred_strings.get(entity_str.lower(), ''))
            stream.write('  %s --> %f   Gold: %s' % (
                entity_str,
                probs[i][j],
                ('X' if batch_entity_ixes[i][j] == correct_candidate else ' ')
            ))
            for (lbl, ix) in pred_ixes:
                stream.write('  %s: %s' % (
                    lbl,
                    ('X' if batch_entity_ixes[i][j] == ix else ' ')
                ))
            stream.write('\n')

def prepSample(mention, ent_ixer, ctx_ixer, window_size, max_mention_size, max_num_entities, secondary_ent_ixer=None):
    ctx_window_ixes, ctx_window_mask = [], []
    mention_ixes, mention_mask = [], []
    ent_ixes, ent_mask = [], []
    secondary_ent_ixes = []

    # context
    for context_string in [mention.left_context, mention.right_context]:
        words = [s.strip() for s in context_string.lower().split()]
        for i in range(window_size):
            if i >= len(words): ix = -1
            else: ix = ctx_ixer.indexOf(words[i])
            if ix >= 0:
                ctx_window_ixes.append(ix)
                ctx_window_mask.append(1)
            else:
                ctx_window_ixes.append(0)
                ctx_window_mask.append(0)

    # mentions
    words = [s.strip() for s in mention.mention_text.lower().split()]
    for i in range(max_mention_size):
        if i >= len(words): ix = -1
        else: ix = ctx_ixer.indexOf(words[i])
        if ix >= 0:
            mention_ixes.append(ix)
            mention_mask.append(1)
        else:
            mention_ixes.append(0)
            mention_mask.append(0)

    # entities
    for i in range(max_num_entities):
        if i >= len(mention.candidates): ix = -1
        else: ix = ent_ixer.indexOf(mention.candidates[i].strip().lower())
        if ix >= 0:
            ent_ixes.append(ix)
            ent_mask.append(1)
        else:
            ent_ixes.append(0)
            ent_mask.append(0)

        if secondary_ent_ixer is None or i >= len(mention.candidates):
            ix = -1
        elif secondary_ent_ixer:
            ix = secondary_ent_ixer.indexOf(mention.candidates[i].strip().lower())
        if ix >= 0:
            secondary_ent_ixes.append(ix)
        else:
            secondary_ent_ixes.append(0)

    correct_candidate = ent_ixer.indexOf(mention.CUI.strip().lower())

    return (ctx_window_ixes, ctx_window_mask, mention_ixes, mention_mask, ent_ixes, ent_mask, secondary_ent_ixes, correct_candidate, mention)

def readDefinitions(f):
    definitions = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (entity, definition) = [s.strip().lower() for s in line.split('\t')]
            definitions[entity] = definition
    return definitions

def embedDefinitions(definitions, word_embs):
    embedded = pyemblib.Embeddings()
    stops = set([s.lower() for s in stopwords.words('english')])
    for (entity, definition) in definitions.items():
        word_arr = []
        for word in definition.split():
            if word in stops:
                continue
            if word in word_embs:
                word_arr.append(word_embs[word])
        if len(word_arr) > 0:
            word_arr = np.array(word_arr)
            embedded[entity] = np.mean(word_arr, axis=0)
    return embedded

if __name__=='__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog MENTIONS [options] --entities=ENTITY_FILE --ctxs=CTX_FILE',
                description='Runs the LogLinearLinker model using the embeddings in ENTITY_FILE and CTX_FILE'
                            ' on the mentions in MENTIONS.')
        parser.add_option('--entities', dest='entity_embf',
                help='entity embedding file (required)')
        parser.add_option('--words', dest='word_embf',
                help='word embedding file (required if using --entity-definitions')
        parser.add_option('--entity-definitions', dest='entity_defnf',
                help='file with extended entity definitions, to use as secondary'
                     ' entity representations')
        parser.add_option('--entity-dual-projections', dest='entity_dualf',
                help='file to write dual projections of entities to')
        parser.add_option('--ctxs', dest='ctx_embf',
                help='context embedding file (required)')
        parser.add_option('--minibatch-size', dest='minibatch_size',
                help='batch size for execution (default: %default)',
                type='int', default=5)
        parser.add_option('--predictions', dest='preds_file',
                help='file to write prediction details to')
        parser.add_option('--combo-method', dest='combo_method',
                help='combination method for primary/secondary entity representations (default: %default)',
                type='choice', choices=[EntityComboMethods.Sum, EntityComboMethods.Multiply], default=EntityComboMethods.Sum)
        parser.add_option('--use-mention', dest='using_mention',
                help='flag to use mention text in model (default: off)',
                action='store_true', default=False)
        parser.add_option('--tab-separated', dest='tab_separated',
                help='flag that we\'re reading tab-separated embeddings file',
                action='store_true', default=False)
        parser.add_option('--strings', dest='stringsf',
                help='file listing preferred strings for entities')
        parser.add_option('--polysemy', dest='polysemyf',
                help='file listing corpus polysemys for entities')
        parser.add_option('-l', '--logfile', dest='logfile',
                help=str.format('name of file to write log contents to (empty for stdout)'),
                default=None)
        (options, args) = parser.parse_args()
        if len(args) != 1 or (not options.entity_embf) or (not options.ctx_embf) or (options.entity_defnf and not options.word_embf):
            parser.print_help()
            exit()
        (mentionf,) = args
        return mentionf, options

    mentionf, options = _cli()
    log.start(logfile=options.logfile, stdout_also=True)

    if options.tab_separated:
        sep = '\t'
    else:
        sep = ' '

    t_sub = log.startTimer('Reading entity embeddings from %s...' % options.entity_embf, newline=False)
    entity_embeds = pyemblib.read(options.entity_embf, separator=sep)
    log.stopTimer(t_sub, message='Read %d embeddings ({0:.2f}s)' % len(entity_embeds))

    t_sub = log.startTimer('Reading context embeddings from %s...' % options.ctx_embf, newline=False)
    ctx_embeds = pyemblib.read(options.ctx_embf, separator=sep)
    log.stopTimer(t_sub, message='Read %d embeddings ({0:.2f}s)' % len(ctx_embeds))

    if options.entity_defnf:
        t_sub = log.startTimer('Reading word embeddings from %s...' % options.word_embf, newline=False)
        #word_embeds = pyemblib.read(options.word_embf)
        word_embeds = ctx_embeds
        log.stopTimer(t_sub, message='Read %d embeddings ({0:.2f}s)' % len(word_embeds))

        t_sub = log.startTimer('Reading entity definitions from %s...' % options.entity_defnf, newline=False)
        definitions = readDefinitions(options.entity_defnf)
        log.stopTimer(t_sub, message='Read %d definitions ({0:.2f}s)' % len(definitions))

        log.write('Constructing entity definition representations...')
        entity_defn_embeds = embedDefinitions(definitions, word_embeds)
        #del(word_embeds)
        log.writeln('Embedded %d entity definitions.' % len(entity_defn_embeds))

        if options.entity_dualf:
            dual_embeds = pyemblib.Embeddings()
            for (k, v) in entity_defn_embeds.items():
                if k in entity_embeds:
                    dual_embeds[k] = np.concatenate([entity_embeds[k], v])
            log.writeln('Writing both versions of entity embeddings to %s...' % options.entity_dualf)
            pyemblib.write(dual_embeds, options.entity_dualf)
            log.writeln('Wrote %d dual embeddings.' % len(dual_embeds))
    else:
        entity_defn_embeds = None

    if options.stringsf:
        t_sub = log.startTimer('Reading preferred strings from %s...' % options.stringsf)
        preferred_strings = readPreferredStrings(options.stringsf)
        log.stopTimer(t_sub, message='Read %d strings ({0:.2f}s)' % len(preferred_strings))
    else:
        preferred_strings = None

    if options.polysemyf:
        t_sub = log.startTimer('Reading corpus polysemy from %s...' % options.polysemyf)
        polysemy = readPolysemy(options.polysemyf)
        log.stopTimer(t_sub, message='Read %d polysemys ({0:.2f}s)' % len(polysemy))
    else:
        polysemy = None

    t_sub = log.startTimer('Reading mentions from %s...' % mentionf, newline=False)
    mentions = mention_file.read(mentionf)
    log.stopTimer(t_sub, message='Read %d mentions ({0:.2f}s)' % len(mentions))

    if options.preds_file:
        preds_stream = open(options.preds_file, 'w')
        preds_stream_detailed = {
            'all': codecs.open('%s.detailed' % options.preds_file, 'w', 'utf-8')
        }
        if entity_defn_embeds:
            preds_stream_detailed['both_correct'] = codecs.open('%s.detailed.dual.both_correct' % options.preds_file, 'w', 'utf-8')
            preds_stream_detailed['entity_only_correct'] = codecs.open('%s.detailed.dual.entity_only_correct' % options.preds_file, 'w', 'utf-8')
            preds_stream_detailed['secondary_only_correct'] = codecs.open('%s.detailed.dual.secondary_only_correct' % options.preds_file, 'w', 'utf-8')
            preds_stream_detailed['both_wrong'] = codecs.open('%s.detailed.dual.both_wrong' % options.preds_file, 'w', 'utf-8')
            preds_stream_detailed['ent_joint_help'] = codecs.open('%s.detailed.joint.ent_joint_help' % options.preds_file, 'w', 'utf-8')
            preds_stream_detailed['ent_joint_hurt'] = codecs.open('%s.detailed.joint.ent_joint_hurt' % options.preds_file, 'w', 'utf-8')
            preds_stream_detailed['sec_joint_help'] = codecs.open('%s.detailed.joint.sec_joint_help' % options.preds_file, 'w', 'utf-8')
            preds_stream_detailed['sec_joint_hurt'] = codecs.open('%s.detailed.joint.sec_joint_hurt' % options.preds_file, 'w', 'utf-8')
        else:
            preds_stream_detailed['entity_correct'] = codecs.open('%s.detailed.entity_only.correct' % options.preds_file, 'w', 'utf-8')
            preds_stream_detailed['entity_wrong'] = codecs.open('%s.detailed.entity_only.wrong' % options.preds_file, 'w', 'utf-8')
        if options.polysemyf:
            preds_stream_polysemy = open('%s.polysemy' % options.preds_file, 'w')
        else:
            preds_stream_polysemy = None
    else:
        preds_stream = None
        preds_stream_detailed = None
        preds_stream_polysemy = None

    correct, total, oracle = runModel(mentions, entity_embeds, ctx_embeds, options.minibatch_size,
        preds_stream, debug=False, secondary_entity_embeds=entity_defn_embeds,
        entity_combo_method=options.combo_method, using_mention=options.using_mention,
        preds_file_detailed=preds_stream_detailed, preferred_strings=preferred_strings,
        preds_file_polysemy=preds_stream_polysemy, polysemy=polysemy)
    lbl = "Joint accuracy" if entity_defn_embeds else "Accuracy"
    print("%s: %.4f (%d/%d)" % (lbl, correct/total, int(correct), total))
    print("Oracle accuracy")
    print("  Entity: %.4f (%d/%d)" % (float(oracle['entity_correct'])/total, oracle['entity_correct'], total))
    if entity_defn_embeds:
        print("  Definitions: %.4f (%d/%d)" % (float(oracle['secondary_correct'])/total, oracle['secondary_correct'], total))
        print("  Oracle: %.4f (%d/%d)" % (float(oracle['oracle_correct'])/total, oracle['oracle_correct'], total))

    if options.preds_file:
        preds_stream.write('Accuracy: %.4f (%d/%d)\n' % (correct/total, int(correct), total))
        preds_stream.write("Oracle accuracy\n")
        preds_stream.write("  Entity: %.4f (%d/%d)\n" % (float(oracle['entity_correct'])/total, oracle['entity_correct'], total))
        if entity_defn_embeds:
            preds_stream.write("  Secondary: %.4f (%d/%d)\n" % (float(oracle['secondary_correct'])/total, oracle['secondary_correct'], total))
            preds_stream.write("  Oracle: %.4f (%d/%d)\n" % (float(oracle['oracle_correct'])/total, oracle['oracle_correct'], total))
        preds_stream.close()

    if preds_stream_detailed:
        for v in preds_stream_detailed.values():
            v.close()
    if preds_stream_polysemy: preds_stream_polysemy.close()
