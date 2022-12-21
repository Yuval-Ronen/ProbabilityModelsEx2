import math
import numpy as np

def lidstoneProbability(c_x, lam, S, X):
    """c_x- number of occurrences of the event in set. S is the set size.
     X= number of different words in set"""
    return float(c_x + lam) / (S + lam * X)

def perplexity(details, lam, S, X, sets, model):
    if lam == 0:
        return float("inf")
    logSum = 0
    for w in sets.validationSet:
        if model == "lidstone":

            p_w = lidstoneProbability(sets.wordsCounter[w], lam, sets.trainingSetSize, details.languageVocabularySize)
            logSum += math.log(p_w)

    return math.exp(-logSum / sets.validationSetSize)



def lidstoneTraining(details, sets):
    details.output[8] = sets.validationSetSize
    details.output[9] = sets.trainingSetSize
    details.output[10] = len(sets.wordsCounter)  # number of different events in the training set
    details.output[11] = sets.wordsCounter[details.inputWord]  # the number of times the event INPUT WORD appears in trainingSet
    details.output[12] = sets.wordsCounter[details.inputWord] / sets.trainingSetSize # MLE based on the training set-no smoothing c(x)\|S|
    # MLE assigns to unseen events if the word ’unseen-word’ is not in the training set
    details.output[13] = sets.wordsCounter["unseen-word"] / sets.trainingSetSize
    # MLE based on the training set using λ = 0.10
    details.output[14] = lidstoneProbability(sets.wordsCounter[details.inputWord], 0.10, sets.trainingSetSize, details.languageVocabularySize)
    details.output[15] = lidstoneProbability(sets.wordsCounter["unseen-word"], 0.10, sets.trainingSetSize, details.languageVocabularySize)

    perplexityDict = {
        lam: perplexity(details, lam, sets.trainingSetSize, details.languageVocabularySize, sets,"lidstone")
        for lam in np.arange(0.0, 2.01, 0.01)}
    details.output[16] = perplexityDict[0.01]  # The perplexity on the validation set using λ = 0.01
    details.output[17] = perplexityDict[0.1]  # The perplexity on the validation set using λ = 0.10
    details.output[18] = perplexityDict[1.0]  # The perplexity on the validation set using λ = 1.00

    valList = list(perplexityDict.values())
    keyList = list(perplexityDict.keys())
    minLambdaIndex = valList.index(min(valList))  # this is the index of the minimal lambda
    valueOfMinLam = valList[minLambdaIndex]  # this is the value of the minimal lambda
    bestLambda = keyList[minLambdaIndex]
    details.output[19] = bestLambda  # The value of λ that i found to minimize the perplexity on the validation set
    details.output[20] = valueOfMinLam  # The minimized perplexity on the validation set using the best value you found for λ


    # details.output[16] = perplexity(details, 0.01, sets.trainingSetSize, details.languageVocabularySize, sets,"lidstone")
    # details.output[17] = perplexity(details, 0.10, sets.trainingSetSize, details.languageVocabularySize, sets,"lidstone")
    # details.output[18] = perplexity(details, 1.00, sets.trainingSetSize, details.languageVocabularySize, sets,"lidstone")
    # details.output[19] =
    # details.output[20] =