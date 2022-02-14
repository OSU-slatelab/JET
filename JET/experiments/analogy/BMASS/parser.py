'''
Parser for BMASS files
'''

import codecs
from . import settings

def _readMultipleEntries(entry):
    entries = []
    cur_entry, in_string = [], False
    for char in entry:
        if char == ',' and not in_string:
            entries.append(''.join(cur_entry))
            cur_entry = []
        else:
            cur_entry.append(char)
            if char == '"': in_string = not in_string
    entries.append(''.join(cur_entry))
    return entries
                
def _parseLine(line, multi_b, multi_d, strings_only, cuis_only):
    if strings_only:
        cui_str = lambda s: s.split(':')[1].strip('"')
    elif cuis_only:
        cui_str = lambda s: s.split(':')[0]
    else:
        cui_str = lambda s: (s.split(':')[0], s.split(':')[1].strip('"'))

    (a,b,c,d) = [s.strip() for s in line.split('\t')]
    a, c = cui_str(a), cui_str(c)

    if multi_b:
        b = [cui_str(b_i) for b_i in _readMultipleEntries(b)]
    else:
        b = cui_str(b)

    if multi_d:
        d = [cui_str(d_i) for d_i in _readMultipleEntries(d)]
    else:
        d = cui_str(d)

    return (a,b,c,d)

def read(analogy_file, setting, strings_only=False, cuis_only=False):
    multi_b = setting == settings.ALL_INFO
    multi_d = setting in [settings.ALL_INFO, settings.MULTI_ANSWER]

    analogies = {}
    with codecs.open(analogy_file, 'r', 'utf-8') as stream:
        cur_relation, cur_analogies = None, []
        for line in stream:
            # relation separators
            if line[0] == '#':
                if cur_relation:
                    analogies[cur_relation] = cur_analogies
                cur_relation = line[2:].strip()
                cur_analogies = []
            # everything else is an analogy
            else:
                analogy = _parseLine(line, multi_b, multi_d, strings_only, cuis_only)
                cur_analogies.append(analogy)
        analogies[cur_relation] = cur_analogies
    return analogies
