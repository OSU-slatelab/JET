'''
Rewrite of corpus tagging
(1) Generate standoff annotations (not original text)
(2) Allow overlapped phrases
'''

import random
import operator
import codecs
import time
import multiprocessing as mp
import dependencies.drgriffis.common.preprocessing as pre
from dependencies.drgriffis.common import pickleio
from dependencies.drgriffis.common import log
from dependencies.drgriffis.common import util

class _SIGNALS:
    HALT = -1

class PartialTermSet:
    '''Used for tracking the set of terms that are current potential
    matches in a line of text.
    '''
    def __init__(self):
        self.terms = {}
    def add(self, pt_term_match):
        '''Add a new PartialTermMatch to the set for tracking'''
        # find a new random identifier
        rand_id = None
        while not rand_id or self.terms.get(rand_id, None) != None:
            rand_id = random.randint(0,100000)
        # assign it to the pt_term_match and store
        pt_term_match.set_id = rand_id
        self.terms[pt_term_match.set_id] = pt_term_match
    def drop(self, pt_term_match):
        '''Stop tracking an unsatisfied PartialTermMatch'''
        del self.terms[pt_term_match.set_id]
    def __iter__(self):
        return iter(list(self.terms.values())) # make sure to copy it

class PartialTermMatch:
    '''Tracks the start index and remaining potential word
    matches to satisfy a known term
    '''
    def __init__(self, start_ix, subtree):
        self.start_ix = start_ix
        self.subtree = subtree
        self.num_tokens = 0
        self.set_id = None
    def increment(self):
        self.num_tokens += 1
    def toCompleteMatch(self, term_id):
        return (
            self.start_ix,
            self.num_tokens,
            term_id
        )

def tokenize(line):
    #return line.strip().lower().split()
    return pre.tokenize(line)

def tagLine(tokens, strmap, cur_word_ix):
    '''
    Inputs:
        tokens       The tokenized line of text to tag
        strmap       Token-wise tree structure mapping strings to term identifiers
        cur_word_ix  Current word offset (used in locating standoff annotations)

    Returns:
        List of (start_ix, end_ix, term_id) triples for this line
    '''

    active_terms = PartialTermSet()
    matched_terms = []

    for token in tokens:
        ## (1) check if this token starts any new terms
        if strmap.get(token, None) != None:
            pt_term_match = PartialTermMatch(cur_word_ix, strmap)
            active_terms.add(pt_term_match)
        
        ## (2) check if this token continues, terminates, or rules out any active terms
        for term in active_terms:
            result = term.subtree.get(token, None)

            # Case 1: rules out continuation
            #         remove the active term
            if result is None:
                active_terms.drop(term)

            # Case 2: terminates a known n-gram
            #         save the term and check for children
            elif type(result) is tuple:
                term.increment()
                (term_str, term_id, children) = result
                matched_terms.append(term.toCompleteMatch(term_id))
                # Case 2.1: no further children
                #           remove the term
                if len(children) == 0:
                    active_terms.drop(term)
                # Case 2.2: has further children (superstrings)
                #           move down the subtree
                else:
                    term.subtree = children

            # Case 3: continues a partial n-gram
            #         move down the subtree
            elif type(result) is dict:
                term.increment()
                term.subtree = result

            else:
                raise Exception("Unexpected element in n-gram tree structure!")

        ## (3) increment the index counter
        cur_word_ix += 1

    # sort match_terms by (start_ix, end_ix)
    matched_terms.sort(key=operator.itemgetter(0,1))
    return matched_terms

def _enqueueLines(corpus, line_q, nthread, max_lines_in_queue):
    with codecs.open(corpus, 'r', 'utf-8') as stream:
        line_ix, cur_word_ix = 0, 0
        for line in stream:
            # lines can be really large, so give the option to limit the
            # line_q's memory usage by capping the number of lines in the
            # queue at once
            if max_lines_in_queue > 0:
                while line_q.qsize() > max_lines_in_queue:
                    time.sleep(0.5)

            tokens = tokenize(line)
            line_q.put( (line_ix, tokens, cur_word_ix) )

            line_ix += 1
            cur_word_ix += len(tokens) + 1   # add 1 token for EOL

    # signal all line processors to halt
    for _ in range(nthread):
        line_q.put(_SIGNALS.HALT)

def _tagLines(line_q, annot_q, strmap):
    while True:
        packet = line_q.get()
        if packet == _SIGNALS.HALT: break
        (line_ix, tokens, cur_word_ix) = packet
        annotations = tagLine(tokens, strmap, cur_word_ix)
        annot_q.put( (line_ix, annotations) )

def _writeAnnotations(outfile, annot_q):
    '''(Multithreaded) Writes annotations to a file, one per line.

    Annotation format is:
       <offset> <# of tokens> <term ID>

    Where offset is the number of tokens from the start of the immediately
    preceding annotation.  (Reduces file size)
    '''
    previous_start_ix = 0
    with codecs.open(outfile, 'w', 'utf-8') as stream:
        write_queue, write_ix, halting = {}, 0, False
        log.track(message='  >> Processed {0:,} lines', writeInterval=100)
        while True:
            packet = annot_q.get()
            if packet == _SIGNALS.HALT: halting = True

            if not halting:
                # store the next tagged line in the priority queue, indexed by position
                (line_ix, annotations) = packet
                write_queue[line_ix] = annotations

            # check if the next position to write has been queued yet
            while not write_queue.get(write_ix, None) is None:
                annotations = write_queue.pop(write_ix)
                for (start_ix, num_tokens, term_id) in annotations:
                    offset = start_ix - previous_start_ix
                    stream.write('%d %d %s\n' % (offset, num_tokens, term_id))
                    previous_start_ix = start_ix
                write_ix += 1
                log.tick()

            # make sure that we've cleared out all of the write queue
            if halting:
                if len(write_queue) > 0:
                    print(write_queue)
                    raise Exception("Line missing: ordered write queue is not empty!")
                break

    log.flushTracker()

def tagCorpus(corpusf, strmap, outfile, nthread, max_lines_in_queue=0):
    '''
    '''

    line_q, annot_q, sig_q = mp.Queue(), mp.Queue(), mp.Queue()

    enqueuer = mp.Process(target=_enqueueLines, args=(corpusf, line_q, nthread-2, max_lines_in_queue))
    taggers = [mp.Process(target=_tagLines, args=(line_q, annot_q, strmap)) for _ in range(nthread-2)]
    writer = mp.Process(target=_writeAnnotations, args=(outfile, annot_q))

    # kick off all the threads
    enqueuer.start()
    writer.start()
    util.parallelExecute(taggers)
    # notify the writer that all lines have been processed
    annot_q.put(_SIGNALS.HALT)
    # close up remaining threads
    enqueuer.join()
    writer.join()



if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog CORPUS STRINGMAP OUTFILE')
        parser.add_option('-t', '--threads', dest='threads',
                help='number of threads to use for tagging the corpus (minimum 3, default %default)',
                type='int', default=3)
        parser.add_option('--max-lines', dest='maxlines',
                help='maximum number of lines from the corpus to hold in memory at once (default no maximum)',
                type='int', default=0)
        parser.add_option('-l', '--logfile', dest='logfile',
                help=str.format('name of file to write log contents to (empty for stdout)'),
                default=None)
        (options, args) = parser.parse_args()
        if len(args) != 3 or options.threads < 3:
            parser.print_help()
            exit()
        corpus = args[0]
        stringmap = args[1]
        outfile = args[2]
        return corpus, stringmap, outfile, options.threads, options.maxlines, options.logfile

    def _logstart(args):
        (corpus, stringmap, outfile, nthread, maxlines) = args
        log.writeln('Tagging strings in corpus.')
        log.writeln('  Corpus in: %s' % corpus)
        log.writeln('  String map in: %s' % stringmap)
        log.writeln('  Writing tagged corpus to: %s' % outfile)
        log.writeln('  Tagging with # of threads: %d' % nthread)
        log.writeln('  Capping line queue size at: %s' % ("unlimited" if maxlines <= 0 else str(maxlines)))
        log.writeln()

    (corpus, stringmap, outfile, nthread, maxlines, logfile) = args = _cli()
    log.start(logfile=logfile, message=_logstart, args=args[:-1], stdout_also=True)

    t_sub = log.startTimer('Loading pickled strings map...')
    ngrams = pickleio.read(stringmap)
    log.stopTimer(t_sub, message='Done in {0:.2f}s')
    
    t_sub = log.startTimer('Tagging corpus...')
    tagCorpus(corpus, ngrams, outfile, nthread, max_lines_in_queue=maxlines)
    log.stopTimer(t_sub, message='Done in {0:.2f}s')
