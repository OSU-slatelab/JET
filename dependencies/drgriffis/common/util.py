'''
Utility functions for use in any situation.

aka All the stuff I'm tired of copy-pasting :P
'''
import random
import codecs
import re
from drgriffis.common.replacer import replacer

def laxIncrement(dct, key, by=1):
    if not dct.get(key):
        dct[key] = by
    else:
        dct[key] += by

def expectKey(dct, key, valIfNew):
    if dct.get(key, None) is None:
        dct[key] = valIfNew
        return False
    else:
        return True

def dump(fname, contents, encoding='ascii'):
    f = codecs.open(fname, 'w', encoding)
    f.write(contents)
    f.close()

def readlines(fname, encoding='ascii'):
    f = codecs.open(fname, 'r', encoding)
    lns = f.readlines()
    f.close()
    return lns

def readCSV(fname, sep=',', readas=str, encoding='ascii'):
    lns = readlines(fname, encoding=encoding)
    return [[readas(c.strip()) for c in row.split(sep)] for row in lns]

def writeCSV(fname, csv, sep=',', encoding='ascii', useUnicode=False, headers=None):
    '''
    If useUnicode is True and encoding is unspecified, will default to UTF-8 encoding.
    '''
    # add the headers if specified
    if type(headers) in (list, tuple):
        new_csv = [headers]
        new_csv.extend(csv)
        csv = new_csv
    if useUnicode:
        if encoding == 'ascii': encoding = 'utf-8'
        writeas = unicode
    else:
        writeas = str
    dump(fname, toCSV(csv, sep, writeas=writeas), encoding=encoding)

def writeList(fname, data, encoding='ascii'):
    '''Write a list of objects to a file, one per line
    '''
    dump(fname, '\n'.join([str(s) for s in data]), encoding=encoding)

def readList(fname, encoding='ascii', readas=str):
    '''Read a list of objects from a file, one per line
    '''
    data = []
    with codecs.open(fname, 'r', encoding=encoding) as stream:
        for line in stream:
            data.append(readas(line.strip()))
    return data

def toCSV(data, sep=',', writeas=str):
    return '\n'.join([sep.join([writeas(c) for c in row]) for row in data])

def bitflag(bln):
    if bln: return 1
    else: return 0

def transformListToDict(lst, tfrm):
    out = {}
    for i in lst:
        key, val = tfrm(i)
        out[key] = val
    return out

def transformDict(dct, tfrm):
    out = {}
    for k in dct.keys():
        key, val = tfrm(k, dct[k])
        out[key] = val
    return out

def reverseDict(dct, allow_collisions=False):
    if not allow_collisions:
        reverse = lambda key, val: (val, key)
        return transformDict(dct, reverse)
    else:
        rev = {}
        for (k, v) in dct.items():
            if rev.get(v, None) is None: rev[v] = []
            rev[v].append(k)
        return rev

def replace(text, repls):
    pattern = replacer.prepare(repls)
    return replacer.apply(pattern, text)

def prepareForParallel(data, threads, data_only=False):
    '''Chunks list of data into disjoint subsets for each thread
    to process.

    Parameters:
        data    :: the list of data to split among threads
        threads :: the number of threads to split for
    '''
    perthread = int(len(data) / threads)
    threadchunks = []
    for i in range(threads):
        startix, endix = (i*perthread), ((i+1)*perthread)
        # first N-1 threads handle equally-sized chunks of data
        if i < threads-1:
            endix = (i+1)*perthread
            threadchunks.append((startix, data[startix:endix]))
        # last thread handles remainder of data
        else:
            threadchunks.append((startix, data[startix:]))
    if data_only: return [d for (ix, d) in threadchunks]
    else: return threadchunks

def parallelExecute(processes):
    '''Takes instances of multiprocessing.Process, starts them all executing,
    and returns when they finish.
    '''
    # start all the threads running...
    for p in processes:
        p.start()
    # ...and wait for all of them to finish
    for p in processes:
        p.join()

def coinflip():
    '''Return True with probability 1/2
    '''
    return rollDie(n=2)

def rollDie(n):
    '''Roll an N-sided die and return True with probability 1/N
    '''
    return random.choice(range(n)) == 0

def matchesRegex(regex, string):
    '''Returns Boolean indicating if the input regex found a positive (non-zero)
    match in the input string.
    '''
    mtch = re.match(regex, string)
    return mtch != None and mtch.span() != (0,0)

def flatten(*arr):
    '''Given one or more N-dimensional objects (N can vary), returns 1-dimensional
    list of the contained objects.
    '''
    results = []
    for el in arr:
        if type(el) == list or type(el) == tuple or type(el) == type({}.values()): results.extend(flatten(*el))
        else: results.append(el)
    return results

def XMLAttribute(attr_name, xml):
    '''Given an XML string and an attribute name, return the value of that attribute
    '''
    match = re.findall('%s=".+"' % attr_name, xml)
    if len(match) != 1:
        raise KeyError('Attribute name "%s" not found/not unique' % attr_name)
    match = match[0]

    opn = match.index('"')
    cls = match[opn+1:].index('"')
    return match[opn+1:opn+cls+1]

def XMLValue(xml):
    '''Given an XML string, returns the value of the tag (if present)
    '''
    match = re.findall(r'>.+</', xml)
    if len(match) != 1:
        return None
    else:
        return match[0][1:-2]

def sortFrequencyDictionary(freq_dict, descending=True):
    '''Returns a list of (item, frequency) pairs, in sorted order
    '''
    mapper = {}
    for (item, freq) in freq_dict.items():
        if mapper.get(freq, None) is None:
            mapper[freq] = []
        mapper[freq].append(item)

    freqs = list(mapper.keys())
    freqs.sort()
    if descending: freqs.reverse()

    sorted_pairs = []
    for freq in freqs:
        sorted_pairs.extend([(item, freq) for item in mapper[freq]])
    return sorted_pairs
