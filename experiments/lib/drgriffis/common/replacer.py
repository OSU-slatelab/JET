import re

class replacer:
    """
    Thanks to: http://stackoverflow.com/questions/6116978/python-replace-multiple-strings
    """
    @staticmethod
    def prepare(rep, onlyAtEnds=False, multiOccur=True):
        '''Compiles and returns a regex matching the input list of strings to replace
        Note: returns two values wrapped as one; can feed tuple directly into apply
        '''
        if type(rep) == list:
            rep = {key: '' for key in rep}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        if onlyAtEnds:
            if multiOccur:
                expr = str.format("^[{0}]+|[{0}]+$", ''.join(rep.keys()))
            else:
                expr = str.format("{0}|{1}",
                    "|".join(['^%s' % key for key in rep.keys()]),
                    "|".join(['%s$' % key for key in rep.keys()])
                )
        else:
            expr = "|".join(rep.keys())
        pattern = re.compile(expr)
        return (pattern, rep)

    @staticmethod
    def apply(pattern_rep, text):
        '''Uses a compiled pattern from .prepare() to replace all instances of desired strings in text
        '''
        pattern, rep = pattern_rep
        return pattern.sub(lambda m: rep[re.escape(m.group(0))], str(text))

    @staticmethod
    def remove(pattern_rep, text):
        pattern, _ = pattern_rep
        return pattern.sub('', str(text))

    @staticmethod
    def suball(pattern_rep, sub_with, text):
        pattern, _ = pattern_rep
        return pattern.sub(sub_with, str(text))
