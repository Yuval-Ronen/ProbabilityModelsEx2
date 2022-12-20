import sys

def myInit(details, output):
    output[1] = details.developmentSetFileName
    output[2] = details.testSetFileName
    output[3] = details.inputWord
    output[4] = details.outputFileName
    output[5] = details.languageVocabularySize
    output[6] = 1 / details.languageVocabularySize
    print(output)
