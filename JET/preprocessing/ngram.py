_ID_PATTERN_='{0}'

class NGramMapPrinter:
    @staticmethod
    def prn(ngmap, root=True):
        if root: output = ['NGramMap: {']
        else: output = []
        for (key, val) in ngmap.items():
            if type(val) == tuple:
                output.append('  %s -> %s' % (val[0], str(val[1])))
                children = val[2]
            else:
                children = val
            if len(children) > 0:
                output.append(NGramMapPrinter.prn(children, root=False))
        if root: output.append('}')

        if not root:
            return '\n'.join(output)
        else:
            print('\n'.join(output))

class NGramMapper:
    
    def __init__(self):
        self.ngrams = {}
        self._id_counter = 0

    def add(self, tokens, mp=None, prevTokens=[], use_collapsed_string=False):
        if mp == None:
            mp = self.ngrams

        if len(tokens) > 0:
            head, tail = tokens[0], tokens[1:]

            # if we're looking at the final token, store it as an (ngram, ID, children) tuple
            if len(tail) == 0:
                allTokens = [t for t in prevTokens]
                allTokens.append(head)
                node = (
                    ' '.join(allTokens),
                    str.format(_ID_PATTERN_,
                        (self._next_id() if not use_collapsed_string else ''.join(allTokens))
                    ),
                    {}
                )
            # otherwise, just store the dictionary of children
            else:
                node = {}

            mappedVal = mp.get(head)
            # if this token is unmapped, save node to it
            if mappedVal == None:
                mp[head] = node
            # if the token is mapped, copy over its children and save the node
            elif mappedVal != None and type(mappedVal) == dict:
                # if the node is a complete n-gram
                if type(node) == tuple:
                    _tmp = list(node)        # cast node tuple to list to make it mutable
                    _tmp[2] = mappedVal      # save the children
                    mp[head] = tuple(_tmp)   # recast to tuple
                # otherwise, just ignore the new node
                else:
                    pass

            # finally, if there are further tokens, recurse on those
            if len(tail) > 0:
                newPrevTokens = [t for t in prevTokens]
                newPrevTokens.append(head)

                return self.add(tail, mp=self._children(mp[head]), prevTokens=newPrevTokens, use_collapsed_string=use_collapsed_string)
            else:
                (_, ID, _) = mp[head]
                return ID

    def _next_id(self):
        self._id_counter += 1
        return self._id_counter

    def _children(self, node):
        if type(node) == dict:
            return node
        elif type(node) == tuple:
            return node[2]
