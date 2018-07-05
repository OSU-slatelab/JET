'''
Abstracted access to embedding for a target concept/phrase.

If using term baseline with word backoff, handles that access under the hood
'''

import embedding_methods as em
import numpy as np

class EmbeddingWrapper:

    _embeds = None
    _embed_vocab = None
    _embed_array = None
    _backoff_embeds = None
    
    def __init__(self, repr_method, embeds, backoff_embeds=None, indexed=False):
        self.repr_method = repr_method
        self.indexed = indexed

        self._backoff_embeds = backoff_embeds

        # if enabling indexed access, split embeddings dictionary into immutable vocabulary and array
        if indexed:
            self._embed_vocab = tuple(embeds.keys())
            self._embed_vocab_indices = { self._embed_vocab[i]:i for i in range(len(self._embed_vocab)) }
        self._embeds = embeds

    def knows(self, item):
        if self.repr_method == em.WORD:
            return np.sum([self._embeds.get(t, None) != None for t in item.split()]) > 0

        elif self.repr_method == em.TERM:
            return self._embeds.get(item, None) != None \
                or (
                    self._backoff_embeds != None 
                    and np.sum([self._backoff_embeds.get(t, None) != None for t in item.split()]) > 0
                   )

        else:
            return not self._embeds.get(item, None) is None

    def index(self, item):
        if self.indexed:
            try:
                return self._embed_vocab_indices[item]
            except (KeyError, ValueError):
                return -1
        else:
            raise NotImplemented('index() only supported with an indexed %s' % self.__class__.__name__)

    def indexToTerm(self, ix):
        if self.indexed:
            return self._embed_vocab[ix]
        else:
            return NotImplemented

    def asArray(self):
        if self.indexed:
            return np.array([self._embeds[v] for v in self._embed_vocab])
        else:
            raise NotImplemented('asArray() only supported with an indexed %s' % self.__class__.__name__)

    def __getitem__(self, item):
        if not self.indexed or (not type(item) is int):
            # word baseline: use average embedding of tokens in string
            if self.repr_method == em.WORD:
                return self._laxTokenAverage(item, self._embeds)

            # string baseline: use string embedding if known, else back off to word baseline
            elif self.repr_method == em.TERM:
                if self._embeds.get(item, None) != None: return self._embeds[item]
                else:
                    return self._laxTokenAverage(item, self._backoff_embeds)

            # concept method
            elif not self._embeds is None:
                return self._embeds[item]

            else:
                raise KeyError
        else:
            return self._embeds[self._embed_vocab[item]]

    def _laxTokenAverage(self, item, embeds):
        token_embeds = []
        for t in item.split():
            if not embeds.get(t, None) is None: token_embeds.append(embeds[t])
        if len(token_embeds) > 0:
            return np.mean(token_embeds, axis=0)
        else: raise KeyError

    def __len__(self):
        return len(self._embeds)
