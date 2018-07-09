from ... import mention_file
from drgriffis.common import preprocessing, log

def getContextWords(context_full, window_size, word_filter):
    i, context = 0, []
    while len(context) < window_size and i < len(context_full):
        token = context_full[i]
        if word_filter(token): context.append(token)
        i += 1
    if len(context) < window_size:
        context.append('[%PADDING%]')
    return context

def getSingleMention(instance, window_size, word_filter, label_set):
    '''Gets the following features:
        1) window_size left context words
        2) window_size right context words
        3) mention words
    '''
    # left contexts
    left_context_full = preprocessing.tokenize(instance.full_text[:instance.begin])
    left_context_full.reverse()
    left_context = getContextWords(left_context_full, window_size, word_filter)
    left_context.reverse() # put it back into document order

    # right contexts
    right_context_full = preprocessing.tokenize(instance.full_text[instance.end:])
    right_context = getContextWords(right_context_full, window_size, word_filter)

    # mention embed
    mention_tokens = preprocessing.tokenize(instance.text)

    return mention_file.Mention(
        instance.CUI.lower(),
        ' '.join(mention_tokens),
        ' '.join(left_context),
        ' '.join(right_context),
        label_set
    )

def getAllMentions(dataset, window_size, word_filter, concept_filter, log=log):
    samples = []
    log.track(message='  >> Extracted features from {0}/%d documents...' % len(dataset), writeInterval=1)
    for ambig in dataset:
        for instance in ambig.instances:
            if concept_filter(instance.CUI):
                samples.append(getSingleMention(instance, window_size, word_filter, ambig.labels))
        log.tick()
    log.writeln()
    return samples
