
import re
import codecs
from .models import EntityMention, AmbiguitySet
from drgriffis.common import util

_relation_regex = re.compile(r'^@RELATION C[0-9]*')
_class_regex = re.compile(r'^@ATTRIBUTE class {M[0-9]')
_data_regex = re.compile(r'^@DATA')

class _LookingFor:
    Relation = 0
    Classes = 1
    DataHeader = 2
    Data = 3

def parseFile(f):
    '''
    '''
    mentions = []
    with codecs.open(f, 'r', 'latin-1') as stream:
        looking_for = _LookingFor.Relation

        for line in stream:
            # force to utf-8 encoding
            line = line.encode('utf-8').decode('utf-8')
            if looking_for == _LookingFor.Relation:
                if util.matchesRegex(_relation_regex, line):
                    chunks = line.split()
                    cuis = chunks[1].split('_')

                    looking_for = _LookingFor.Classes

            elif looking_for == _LookingFor.Classes:
                if util.matchesRegex(_class_regex, line):
                    chunks = line.split()
                    class_list = [s.strip() for s in ''.join(chunks[2:]).strip('{}').split(',')]
                    class_map = { class_list[i]: cuis[i] for i in range(len(class_list)) }

                    looking_for = _LookingFor.DataHeader

            elif looking_for == _LookingFor.DataHeader:
                if util.matchesRegex(_data_regex, line):
                    looking_for = _LookingFor.Data

            else:
                # pull the ID
                first_comma = line.index(',')
                row_id = line[:first_comma]
                # pull the text
                ix = first_comma
                while True:
                    try: ix = (ix+1) + line[ix+1:].index(',')
                    except ValueError: break
                text = line[first_comma+1:ix].strip('"')
                # pull the class label
                label = class_map[line[ix+1:].strip()]

                # find the bounds of the mention
                mention_start = text.index('<e>')
                mention_end = mention_start + text[mention_start+3:].index('</e>')

                text = text[:mention_start] \
                    + text[mention_start+3:mention_end+3] \
                    + text[mention_end+7:]

                mention = EntityMention(label, text, mention_start, mention_end)
                mentions.append(mention)

    ambig_set = AmbiguitySet(set(class_map.values()), mentions)
    return ambig_set

def _test():
    f = '/u/drspeech/data/NLM-WSD/MSHCorpus/TPA_pmids_tagged.arff'
    p = parseFile(f)
    return p
