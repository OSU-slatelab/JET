'''
Parser for Google analogy dataset
'''

import codecs
                
def _parseLine(line, to_lower=False):
    if to_lower:
        line = line.lower()
    (a,b,c,d) = [s.strip() for s in line.split()]
    return (a,b,c,d)

def read(analogy_file, to_lower=False):
    analogies = {}
    with codecs.open(analogy_file, 'r', 'utf-8') as stream:
        cur_relation, cur_analogies = None, []
        for line in stream:
            # relation separators
            if line[0] == ':':
                if cur_relation:
                    analogies[cur_relation] = cur_analogies
                cur_relation = line[2:].strip()
                cur_analogies = []
            # everything else is an analogy
            else:
                analogy = _parseLine(line, to_lower=to_lower)
                cur_analogies.append(analogy)
        analogies[cur_relation] = cur_analogies
    return analogies
