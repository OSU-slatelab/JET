import spacy
import dependencies.drgriffis.common.preprocessing
import optparse

class Tokenizer:
    def __init__(self, options):
        pass

    def tokenize(self, string):
        return NotImplemented

class PreTokenizedTokenizer(Tokenizer):
    def tokenize(self, string):
        tokens = [s.strip() for s in string.split()]
        return tokens

class drgriffisTokenizer(Tokenizer):
    def tokenize(self, string):
        return dependencies.drgriffis.common.preprocessing.tokenize(string)

class SpacyTokenizer(Tokenizer):
    def __init__(self, options):
        self.nlp = spacy.load(options.tokenizer_spacy_model)

    def tokenize(self, string):
        spacy_output = self.nlp(string)
        return [
            str(t)
                for t in spacy_output
        ]

class CLI:
    PreTokenized = 'PreTokenized'
    drgriffis = 'drgriffis'
    Spacy = 'Spacy'

    @staticmethod
    def listTokenizerOptions():
        return [
            CLI.PreTokenized,
            CLI.drgriffis,
            CLI.Spacy
        ]

    @staticmethod
    def tokenizerDefault():
        return CLI.PreTokenized

    @staticmethod
    def initializeTokenizer(options):
        if options.tokenizer_type == CLI.PreTokenized:
            return PreTokenizedTokenizer(options)
        elif options.tokenizer_type == CLI.drgriffisTokenizer:
            return drgriffisTokenizer(options)
        elif options.tokenizer_type == CLI.Spacy:
            return SpacyTokenizer(options)
        else:
            raise KeyError('Tokenizer type "%s" not known' % options.tokenizer_type)

    @staticmethod
    def addOptions(parser):
        group = optparse.OptionGroup(parser, 'Tokenization arguments')
        group.add_option('--tokenizer', dest='tokenizer_type',
            type='choice', choices=CLI.listTokenizerOptions(),
            default=CLI.tokenizerDefault(),
            help='Tokenizer to use')
        group.add_option('--spacy-model', dest='tokenizer_spacy_model',
            default='en_core_web_sm',
            help='SpaCy model to load')
        parser.add_option_group(group)

    @staticmethod
    def logOptions(options):
        return [
            ('Tokenizer', options.tokenizer_type),
            ('SpaCy model', 'N/A' if options.tokenizer_type != SpacyTokenizer else options.tokenizer_spacy_model),
        ]
