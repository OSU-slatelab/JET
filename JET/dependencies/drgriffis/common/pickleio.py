'''
I/O convenience methods for Pickle files
'''

import gzip, pickle

def read(fname):
    '''Return the object pickled in fname
    '''
    hook = gzip.open(fname, 'rb')
    data = pickle.load(hook)
    hook.close()
    return data

def write(data, fname):
    '''Pickle object data to fname
    '''
    hook = gzip.open(fname, 'wb')
    pickle.dump(data, hook)
    hook.close()
