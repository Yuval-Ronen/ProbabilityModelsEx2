import math
import sys
from collections import Counter

from getAllwords import getAllWords
from myInit import myInit
from LidstoneModelTraining import lidstoneTraining



class Details():
    def __init__(self):
        self.developmentSetFileName = sys.argv[1]
        self.testSetFileName = sys.argv[2]
        self.inputWord = sys.argv[3]
        self.outputFileName = sys.argv[4]
        self.languageVocabularySize = 300000
        self.output = [0 for i in range(30)]

class Sets():
    def __init__(self, trainingSetSize, validationSetSize):
        self.trainingSetSize = trainingSetSize
        self.validationSetSize = validationSetSize
        self.trainingSet = allWordsInDev[:trainingSetSize]
        self.validationSet = allWordsInDev[trainingSetSize:]
        self.wordsCounter = Counter(self.trainingSet)
        self.valCounter = Counter(self.validationSet)

def writeToOutput():
    output = open(details.outputFileName, 'w')
    # TODO: add name and id boaz
    # output.write("#Students Yuval Ronen 205380132")
    for i in range(1, 30):
        output.write("\n" + "#Output" + str(i) + " " + str(details.output[i]))
    output.close()


if __name__ == '__main__':
    # input:  < development set filename >
    # < test set filename >
    # < INPUT WORD >
    # < output filename >
    details = Details()

    #### 1 Init
    myInit(details)

    #### 2 Development set preprocessing
    allWordsInDev = getAllWords(details.developmentSetFileName)
    devWordsSize = len(allWordsInDev)
    details.output[7] = devWordsSize

    #### 3 Lidstone model training
    trainingSetSize = round(0.9 * devWordsSize)
    validationSetSize = devWordsSize - trainingSetSize
    sets = Sets(trainingSetSize, validationSetSize)
    lidstoneTraining(details, sets)

    #### 4

    #### 5

    #### 6

    #### 7




    print(details.output)
    writeToOutput()
