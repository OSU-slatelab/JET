import codecs
from ... import mention_file
from . import parser
from drgriffis.common import preprocessing, log

def getMention(tokens, mention, window_size, word_filter):
    (start, end, target, candidates) = mention

    # get left contexts
    left_context_full = preprocessing.tokenize(' '.join(tokens[:start]))
    left_context_full.reverse()
    left_context = getContextWords(left_context_full, window_size, word_filter)
    left_context.reverse() # put it back into document order

    # get right contexts
    right_context_full = preprocessing.tokenize(' '.join(tokens[end:]))
    right_context = getContextWords(right_context_full, window_size, word_filter)

    # mention tokens
    mention_tokens = preprocessing.tokenize(' '.join(tokens[start:end]))

    #print('Parsed:')
    #print('  Left context: %s' % (' '.join(left_context)))
    #print('  Mention: %s' % (' '.join(mention_tokens)))
    #print('  Right context: %s' % (' '.join(right_context)))
    return mention_file.Mention(
        target,
        ' '.join(mention_tokens),
        ' '.join(left_context),
        ' '.join(right_context),
        candidates
    )

def getContextWords(context_full, window_size, word_filter):
    i, context = 0, []
    while len(context) < window_size and i < len(context_full):
        token = context_full[i]
        if word_filter(token): context.append(token)
        i += 1
    if len(context) < window_size:
        context.append('[%PADDING%]')
    return context

def getAllMentions(dataset, window_size, word_filter, concept_filter, log=log):
    data = parser.parse()
    samples, skipped = [], 0
    log.track(message='  >> Extracted features for {0}/%d mentions (skipped {1})...' % len(data), writeInterval=1)

    for (tokens, mentions) in data:
        for mention in mentions:
            (start, end, targets, candidates, correct_ix) = mention
            valid, keep_me = False, None
            for c in targets:
                if concept_filter(c):
                    valid = True
                    keep_me = c
                    break
            if valid:
                candidates[correct_ix] = keep_me
                corrected_mention = (start, end, keep_me, candidates)
                samples.append(getMention(tokens, corrected_mention, window_size, word_filter))
            else:
                skipped += 1
            log.tick(skipped)
    log.flushTracker(skipped)

    return samples
