#Yuval Ronen    Boaz Avraham     205380132   203668132
import sys
from collections import Counter

from getAllwords import getAllWords
from myInit import myInit
from LidstoneModelTraining import lidstoneTraining,heldoutTraining



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
        self.trainingSetWordsCounter = Counter(self.trainingSet)
        self.validationSetWordsCounter = Counter(self.validationSet)

def writeToOutput():
    output = open(details.outputFileName, 'w')
    output.write("#Yuval Ronen\tBoaz Avraham\t205380132\t203668132")
    for i in range(1, 30):
        output.write("\n" + "#Output" + str(i) + "\t" + str(details.output[i]))
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
    lidstoneModel = lidstoneTraining(details, sets)

    #### 4 Held out model training
    trainingSetSize = round(0.5 * devWordsSize)
    validationSetSize = devWordsSize - trainingSetSize
    sets = Sets(trainingSetSize, validationSetSize)
    heldoutModel = heldoutTraining(details, sets)
    #### 5
    lidstoneModel.test_model()
    heldoutModel.test_model()
    #### 6 Models evaluation on test set
    allWordsInTest = getAllWords(details.testSetFileName)
    details.output[25] = len(allWordsInTest)
    details.output[26] = lidstoneModel.perplexity(lidstoneModel.best_lamda, allWordsInTest)
    details.output[27] = heldoutModel.perplexity(allWordsInTest)
    details.output[28] = "L" if details.output[26] < details.output[27] else "H"
    output29 = ""
    for r in range(10):
        f_lam = lidstoneModel.probability(r, lidstoneModel.best_lamda) * lidstoneModel.sets.trainingSetSize
        f_h, N_r, t_r = heldoutModel.probability(r, True)
        f_h *= heldoutModel.sets.trainingSetSize
        f_lam = round(f_lam, 5)
        f_h = round(f_h, 5)
        output29 += "\n{}\t{}\t{}\t{}\t{}".format(r, f_lam, f_h, N_r, t_r)

    details.output[29] = output29




    for x in range(len(details.output)):
        print (str(x) +" - "+ str(details.output[x]))

    writeToOutput()
