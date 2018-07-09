'''
I/O wrappers for mention files (used for streaming feature generation)
'''

import codecs

class Mention:
    def __init__(self, CUI, mention_text, left_context, right_context, candidates):
        self.CUI = CUI
        self.mention_text = mention_text
        self.left_context = left_context
        self.right_context = right_context
        self.candidates = candidates

def write(mentions, outf, encoding='utf-8'):
    with codecs.open(outf, 'w', encoding) as stream:
        for m in mentions:
            stream.write('%s\n' % '\t'.join([
                m.mention_text,
                m.left_context,
                m.right_context,
                m.CUI,
                '||'.join(m.candidates)
            ]))

def read(mentionf, encoding='utf-8'):
    mentions = []
    with codecs.open(mentionf, 'r', encoding) as stream:
        for line in stream:
            (
                mention_text,
                left_context,
                right_context,
                CUI,
                candidates
            ) = [s.strip() for s in line.split('\t')]
            candidates = candidates.split('||')
            mentions.append(Mention(
                CUI, mention_text, left_context, right_context, candidates
            ))
    return mentions
