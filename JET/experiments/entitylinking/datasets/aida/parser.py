'''
Parser for AIDA dataset

Requires AIDA CoNLL-YAGO Dataset, downloadable from
   https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/

And PPRforNED AIDA candidates, developed by Maria Pershina; downloadable from
   https://github.com/masha-p/PPRforNED
'''

import codecs
import os

_datafile = '.../AIDA/AIDA-YAGO2-dataset.tsv'
_candidates_dir = '.../PPRforNED/AIDA_candidates'

_leader = 'http://en.wikipedia.org/wiki/'

def parse(collapse_sentences=False):
    documents = []
    current_document, current_document_id, current_sentence = [], -1,  []

    # read every document in, by sentence
    with codecs.open(_datafile, 'r', 'utf-8') as stream:
        for line in stream:
            if len(line.strip()) == 0:
                if len(current_sentence) > 0:
                    current_document.append(current_sentence)
                current_sentence = []
            elif line.split()[0] == '-DOCSTART-':
                if len(current_sentence) > 0:
                    current_document.append(current_sentence)
                if len(current_document) > 0:
                    documents.append((current_document_id, current_document))
                current_document_id = int(line.split()[1].strip('(testab'))
                current_document = []
                current_sentence = []
            else:
                chunks = [s.strip() for s in line.split('\t')]
                current_sentence.append(chunks)
        if len(current_sentence) > 0:
            current_document.append(current_sentence)
        if len(current_document) > 0:
            documents.append((current_document_id, current_document))

    # set up for independent sequence processing
    extracted = []
    for (doc_id, doc_sentences) in documents:
        if doc_id <= 1000:
            candidate_f = os.path.join(_candidates_dir, 'PART_1_1000', str(doc_id))
        else:
            candidate_f = os.path.join(_candidates_dir, 'PART_1001_1393', str(doc_id))
        candidates = parseCandidates(candidate_f)
        #print("File %d" % doc_id)
        #print(candidate_f)
        #print(candidates)

        if collapse_sentences:
            document = []
            for sentence in doc_sentences:
                document.extend(sentence)

            (tokens, mentions, _) = extractMentions(document, candidates, 0, doc_id=doc_id)
            extracted.append((tokens, mentions))
        else:
            start_ix = 0
            for sentence in doc_sentences:
                (tokens, mentions, start_ix) = extractMentions(sentence, candidates, start_ix, doc_id=doc_id)
                if len(mentions) > 0:
                    extracted.append((tokens, mentions))

    return extracted

def extractMentions(labeled_tokens, candidates, start_ix, doc_id=None):
    tokens, mentions = [], []
    current_mention_target, current_mention_targets = None, None
    current_mention_start = -1
    skipping = False

    for i in range(len(labeled_tokens)):
        chunk = labeled_tokens[i]
        if len(candidates) > 0 and start_ix+len(mentions) < len(candidates):
            (next_entity, next_surface_form, next_candidates, next_correct_ix) = candidates[start_ix+len(mentions)]
        else:
            (next_entity, next_surface_form, next_candidates, next_correct_ix) = None, None, None, None
        #print(chunk, current_mention_start)
        # not in an entity at all
        if len(chunk) == 1:
            tokens.append(chunk[0])
            if current_mention_start >= 0:
                if not skipping:
                    mentions.append((current_mention_start, i, current_mention_targets, next_candidates, next_correct_ix))
                current_mention_target = None
                current_mention_start = -1
        else:
            # if it has no Wikipedia form, just use the --NME-- as "wikipedia" stand-in
            if len(chunk) == 4:
                (token, bio, surface_form, wikipedia) = chunk
            # if it does, take that one
            else:
                (token, bio, surface_form, normalized, wikipedia) = chunk[:5]
                wikipedia = cleanUrl(wikipedia)
            # if we're changing entity (since gets repeated for every token)
            if wikipedia != current_mention_target or bio == 'B':
                if current_mention_start >= 0 and not skipping:
                    mentions.append((current_mention_start, i, current_mention_targets, next_candidates, next_correct_ix))
                    if len(candidates) > 0 and start_ix+len(mentions) < len(candidates):
                        (next_entity, next_surface_form, next_candidates, next_correct_ix) = candidates[start_ix+len(mentions)]
                    else:
                        (next_entity, next_surface_form, next_candidates, next_correct_ix) = None, None, None, None
                current_mention_start = i
                current_mention_target = wikipedia
                current_mention_targets = set([wikipedia, next_entity])
                #print("[INFO - Doc %d] Starting %s..." % (doc_id, current_mention_target))
                #print("[INFO - Doc %d]    %s (%d)" % (doc_id, next_entity, len(mentions)))
                if surface_form != next_surface_form:
                    print("[INFO - Doc %d] Saw surface form '%s' Next expected '%s'; skipping" % (doc_id, surface_form, next_surface_form))
                    assert wikipedia == '--NME--' or next_surface_form is None
                    skipping = True
                else:
                    skipping = False
                if wikipedia != next_entity and (not wikipedia == '--NME--'):
                    print("[WARNING - Doc %d] Expected mention target '%s' got '%s'" % (doc_id, next_entity, current_mention_target))
            tokens.append(token)
    if current_mention_start >= 0 and not skipping:
        mentions.append((current_mention_start, i, current_mention_targets, next_candidates, next_correct_ix))

    #print("[INFO] Done. Added %d mentions; new start ix: %d" % (len(mentions), start_ix+len(mentions)))

    return (tokens, mentions, start_ix+len(mentions))

def parseCandidates(f):
    candidates = []
    current_entity, current_candidates, correct_ix = None, [], -1
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            chunks = line.split('\t')
            if chunks[0] == 'ENTITY':
                if len(current_candidates) > 0:
                    current_candidates = list(current_candidates)
                    for j in range(len(current_candidates)):
                        if current_candidates[j] == current_entity:
                            correct_ix = j
                            break
                    candidates.append((current_entity, current_surface_form, current_candidates, correct_ix))
                current_surface_form = chunks[7][9:].strip()
                current_entity = cleanUrl(chunks[8][4:].strip())
                current_candidates = set([current_entity])
            elif chunks[0] == 'CANDIDATE':
                cand_url = cleanUrl(chunks[5][4:].strip())
                current_candidates.add(cand_url)
        if len(current_candidates) > 0:
            current_candidates = list(current_candidates)
            for j in range(len(current_candidates)):
                if current_candidates[j] == current_entity:
                    correct_ix = j
                    break
            candidates.append((current_entity, current_surface_form, current_candidates, correct_ix))
    return candidates

def cleanUrl(url):
    return url[len(_leader):]

if __name__ == '__main__':
    test = parse(collapse_sentences=False)
