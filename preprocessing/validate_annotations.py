'''
Validate that a corpus file and associated term annotations file are correctly
aligned (i.e., offsets between term instances match up with the correct terms).

Assumes both corpus file and term map have been preprocessed.
'''

from types import SimpleNamespace
from hedgepig_logger import log

BUFF_SIZE = 1000
CORPUS_BUFF_SIZE = 100

def readTermMap(f):
    term_map = {}
    with open(f, 'r') as stream:
        for line in stream:
            (term_id, term_str) = [s.strip() for s in line.split('\t')]
            term_map[int(term_id)] = term_str.split()
    return term_map

def parseAnnotation(line):
    if line is None or len(line.strip()) == 0:
        return None
    else:
        chunks = line.split()
        annot = SimpleNamespace(
            offset = int(chunks[0]),
            num_tokens = int(chunks[1]),
            term_id = int(chunks[2]),
            seen_so_far = 0
        )
        return annot

class AnnotationBuffer:
    
    def __init__(self, annot_stream, term_map):
        self.active_up_to = 0
        self.contents = []
        self._stream = annot_stream
        self.term_map = term_map

    def refill(self):
        reading = True
        while reading:
            if len(self.contents) == BUFF_SIZE:
                reading = False
            else:
                annot = parseAnnotation(self._stream.readline())
                if annot is None:
                    reading = False
                else:
                    self.contents.append(annot)

    def pop(self, index):
        self.contents.pop(index)

    def __repr__(self):
        lines = []
        for i in range(min(len(self.contents), max(self.active_up_to, 5)+2)):
            lines.append(
                '  {0} < Offset: {1}  Tokens: {2}/{3}  String: {4} >'.format(
                    ('->' if i < self.active_up_to else '  '),
                    self.contents[i].offset,
                    self.contents[i].seen_so_far,
                    self.contents[i].num_tokens,
                    ' '.join(self.term_map[self.contents[i].term_id])
                )
            )
        return '[ AnnotationBuffer \n{0}\n...\n]'.format(
            '\n'.join(lines)
        )

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, i):
        return self.contents[i]

class TokenBuffer:
    
    def __init__(self, corpus_stream):
        self.active_index = 0
        self.contents = []
        self._stream = corpus_stream
        self.refill()

    def refill(self):
        # fill the buffer up to initial size
        while len(self.contents) < CORPUS_BUFF_SIZE:
            line = self._stream.readline()
            if line is None or len(line) == 0:
                break
            else:
                for t in line.split():
                    self.contents.append(t.strip())
                self.contents.append('<EOL>')

    def shift(self):
        self.active_index += 1
        if self.active_index >= CORPUS_BUFF_SIZE - 20:
            self.contents = self.contents[self.active_index - 20:]
            self.active_index = 20
            self.refill()

    def __repr__(self):
        left = self.contents[:self.active_index]
        right = self.contents[self.active_index+1:]
        return '{0} [[ {1} ]] {2}'.format(
            ' '.join(left),
            self.contents[self.active_index],
            ' '.join(right)
        )

def validate(corpus_f, annot_f, term_map):
    log.track('  >> Validated {0:,} tokens ({1:,} annotations)', writeInterval=100000)
    annots_so_far = 0
    with open(corpus_f, 'r') as corpus_stream, \
         open(annot_f, 'r') as annot_stream:

        annot_buffer = AnnotationBuffer(annot_stream, term_map)
        annot_buffer.refill()

        token_buffer = TokenBuffer(corpus_stream)

        since_last_term = -1
        while token_buffer.active_index < len(token_buffer.contents):
            t = token_buffer.contents[token_buffer.active_index]

            if len(annot_buffer) == 0:
                break

            # start watching any new terms beginning with this token
            since_last_term += 1
            while (
                (annot_buffer.active_up_to < len(annot_buffer))
                and (since_last_term == annot_buffer[annot_buffer.active_up_to].offset)
            ):
                annot_buffer.active_up_to += 1
                since_last_term = 0

            #print(token_buffer)
            #print(annot_buffer)
            #input()

            # for all watched terms, validate that the current word is as expected
            for i in range(annot_buffer.active_up_to):
                annot = annot_buffer[i]
                if annot.num_tokens != len(term_map[annot.term_id]):
                    raise KeyError('Expected term {0} to have {1:,} tokens, annotation has {2:,}'.format(
                        annot.term_id, len(term_map[annot.term_id]), annot.num_tokens
                    ))
                expected_token = term_map[annot.term_id][annot.seen_so_far]
                if expected_token != t:
                    raise ValueError('Expected token "{0}" for term {1} at position {2}, found "{3}"'.format(
                        expected_token, annot.term_id, annot.seen_so_far, t
                    ))
                annot.seen_so_far += 1

            # go back through the watched terms and unwatch any that have been completed
            i = 0
            while i < annot_buffer.active_up_to:
                if annot_buffer[i].seen_so_far == annot_buffer[i].num_tokens:
                    annot_buffer.pop(i)
                    annot_buffer.active_up_to -= 1
                    annots_so_far += 1
                else:
                    i += 1

            annot_buffer.refill()
            token_buffer.shift()
            log.tick(annots_so_far)
    log.flushTracker(annots_so_far)

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser('Usage: %0')
        parser.add_option('-c', '--corpus', dest='corpus_f',
            help='(required) corpus file')
        parser.add_option('-a', '--annotations', dest='annotations_f',
            help='(required) annotations file')
        parser.add_option('-t', '--term-strings', dest='term_strings_f',
            help='(required) file mapping term IDs to strings')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='file to write log messages to')
        (options, args) = parser.parse_args()
        if not options.corpus_f:
            parser.print_help()
            parser.error('Must provide --corpus')
        if not options.annotations_f:
            parser.print_help()
            parser.error('Must provide --annotations')
        if not options.term_strings_f:
            parser.print_help()
            parser.error('Must provide --term-strings')
        return options
    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('Corpus file', options.corpus_f),
        ('Annotations file', options.annotations_f),
        ('Term strings file', options.term_strings_f),
    ], 'JET annotation validation')

    log.writeln('Reading term->strings mapping from %s...' % options.term_strings_f)
    term_map = readTermMap(options.term_strings_f)
    log.writeln('Mapped strings for {0:,} terms.\n'.format(len(term_map)))

    log.writeln('Validating corpus annotations...')
    validate(options.corpus_f, options.annotations_f, term_map)
    log.writeln('Done!\n')

    log.stop()
