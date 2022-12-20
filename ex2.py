import math
import sys
from myInit import myInit


class Details():
    def __init__(self):
        self.developmentSetFileName = sys.argv[1]
        self.testSetFileName = sys.argv[2]
        self.inputWord = sys.argv[3]
        self.outputFileName = sys.argv[4]
        self.languageVocabularySize = 300000


if __name__ == '__main__':
    # input:  < development set filename >
    # < test set filename >
    # < INPUT WORD >
    # < output filename >

    #### 1 Init
    details = Details()
    output = [0 for i in range(30)]
    myInit(details, output)
