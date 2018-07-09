ALL_INFO = 0
MULTI_ANSWER = 1
SINGLE_ANSWER = 2

def name(x):
    if x == ALL_INFO:
        return 'ALL_INFO'
    elif x == 'MULTI_ANSWER':
        return 'MULTI_ANSWER'
    elif x == 'SINGLE_ANSWER':
        return 'SINGLE_ANSWER'
    else:
        return str(x)
