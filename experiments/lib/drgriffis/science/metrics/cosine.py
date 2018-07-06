'''
Implements cosine similarity/distance for two vectors.
'''

import numpy as np

def CosineSimilarity(a,b):
    '''Calculates the cosine similarity between vectors a,b.

    cos(a,b) defined as dot(a,b)/(norm(a)*norm(b))
    '''
    a,b = np.array(a), np.array(b)
    numerator = np.dot(a,b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return numerator/denominator

def CosineDistance(a,b):
    '''Calculates the cosine distance between vectors a,b.

    Distance defined as 1 - cos(a,b).
    '''
    return 1 - CosineSimilarity(a,b)
