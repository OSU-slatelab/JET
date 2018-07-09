class EntityMention:
    def __init__(self, CUI, full_text, begin, end):
        self._CUI = CUI
        self._full_text = full_text
        self._begin = begin
        self._end = end
    
    @property
    def CUI(self):
        return self._CUI

    @property
    def full_text(self):
        return self._full_text

    @property
    def text(self):
        return self._full_text[self._begin:self._end]

    @property
    def begin(self):
        return self._begin
    @property
    def end(self):
        return self._end

class AmbiguitySet:
    def __init__(self, labels, instances):
        self._labels = labels
        self._instances = instances

    def __iter__(self):
        return iter(self._instances)

    @property
    def instances(self):
        return self._instances

    @property
    def labels(self):
        return self._labels
