class Mode:
    Text = 'txt'
    Binary = 'bin'

class Format:
    Word2Vec = 'Word2Vec'
    Glove = 'Glove'

def getFileSize(inf):
    curIx = inf.tell()
    inf.seek(0, 2)  # jump to end of file
    file_size = inf.tell()
    inf.seek(curIx)
    return file_size
