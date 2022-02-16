import spacy
from ..dependencies.drgriffis.common.preprocessing import tokenize as drgriffis_tokenize
import optparse

PreTokenized = 'PreTokenized'
drgriffis = 'drgriffis'
Spacy = 'Spacy'

class Tokenizer:
    def __init__(self, *args, **kwargs):
        pass

    def tokenize(self, string):
        return NotImplemented

    @staticmethod
    def build(tokenizer, *args, **kwargs):
        if tokenizer == PreTokenized:
            return PreTokenizedTokenizer(*args, **kwargs)
        elif tokenizer == drgriffisTokenizer:
            return drgriffisTokenizer(*args, **kwargs)
        elif tokenizer == Spacy:
            return SpacyTokenizer(*args, **kwargs)
        else:
            raise KeyError('Tokenizer type "%s" not known' % tokenizer)

class PreTokenizedTokenizer(Tokenizer):
    def tokenize(self, string):
        tokens = [s.strip() for s in string.split()]
        return tokens

class drgriffisTokenizer(Tokenizer):
    def tokenize(self, string):
        return drgriffis_tokenize(string)

class SpacyTokenizer(Tokenizer):
    def __init__(self, spacy_model='en_core_web_sm'):
        self.nlp = spacy.load(spacy_model)

    def tokenize(self, string):
        spacy_output = self.nlp(string)
        return [
            str(t)
                for t in spacy_output
        ]

class CLI:
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
        return Tokenizer.build(
            tokenizer=options.tokenizer_type,
            spacy_model=options.tokenizer_spacy_model
        )

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
