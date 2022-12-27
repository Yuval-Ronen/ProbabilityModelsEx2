#Yuval Ronen    Boaz Avraham     205380132   203668132

def getAllWords(file):
    """read the file and compose a list of all the words in file
    words can appear more than one time"""
    readFile = open(file, 'r')
    allWords = []
    for line in readFile:
        values = line.split()
        if len(values) != 0 and values[0] != "<TRAIN" and values[0] != "<TEST":
            # we want to skip the lines of the subjects and empty lines
            allWords += values
    return allWords
