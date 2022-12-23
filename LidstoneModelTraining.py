#Students Yuval Ronen 205380132, Boaz Avraham 203668132

import math
from collections import Counter
from collections import defaultdict

import numpy as np



class LidstoneModel:
    def __init__(self, details, sets):
        self.S = sets.trainingSetSize
        self.X = details.languageVocabularySize
        self.details = details
        self.sets = sets
        self.start_training()

    def start_training(self):
        self.perplexityDict = {
            lam: self.perplexity(lam)
            for lam in np.around(np.arange(0.01, 2.01, 0.01), 2)}
        self.best_lamda = min(self.perplexityDict, key=self.perplexityDict.get)

    def probability(self, c_x, lam):
        """c_x- number of occurrences of the event in set. S is the set size.
         X= number of events in set"""
        return float(c_x + lam) / (self.S + lam * self.X)

    def perplexity(self, lam,  test_set = None):
        logSum = 0
        if test_set == None:
            test_set = self.sets.validationSet
        for w in test_set:
            p_w = self.probability(self.sets.trainingSetWordsCounter[w], lam)
            logSum += math.log(p_w)

        return math.exp(-logSum / len(test_set))

    def test_model(self):
        print ("this is the sum of all the probabilites in lidstone model")
        s = sum(self.probability(r,self.best_lamda) for r in self.sets.trainingSetWordsCounter.values())
        s += sum(self.probability(r, self.best_lamda) for r in self.sets.validationSetWordsCounter.values())
        print(s + self.details.languageVocabularySize * self.probability(0,self.best_lamda))


class HeldoutModel:
    def __init__(self, details, sets):
        self.H = sets.validationSetSize
        self.details = details
        self.sets = sets
        self.N_0 = details.languageVocabularySize - len(sets.trainingSetWordsCounter) #number of events that were not in T
        self.count_to_event_dict = defaultdict(list)
        for key, value in self.sets.trainingSetWordsCounter.items():
            self.count_to_event_dict[value].append(key)

        self.count_to_event_dict[0] = [k for k in self.sets.validationSetWordsCounter.keys() if
                      k not in self.sets.trainingSetWordsCounter.keys()]


    def probability(self, r, elaborate = False):
        x_r_events = self.count_to_event_dict[r]
        t_r = sum(self.sets.validationSetWordsCounter.get(k, 0) for k in x_r_events)

        if r == 0:
            N_r = self.N_0
        else:
            N_r = len(x_r_events)

        p_r = t_r / (self.sets.validationSetSize * N_r)

        if elaborate: #returns p(r), N_r, t_r
            return p_r, N_r, t_r
        return p_r


    def perplexity(self, test_set = None):
        if test_set == None:
            test_set = self.sets.validationSet
        logSum = 0
        for w in test_set:
            p_w = self.probability(self.sets.trainingSetWordsCounter[w])
            logSum += math.log(p_w)

        return math.exp(-logSum / len(test_set))

    def test_model(self):
        print ("this is the sum of all the probabilites in heldout model")
        s = sum(self.probability(r) for r in self.count_to_event_dict.keys())
        print(s + self.N_0 * self.probability(0))



def lidstoneTraining(details, sets):
    model = LidstoneModel(details, sets)
    details.output[8] = sets.validationSetSize
    details.output[9] = sets.trainingSetSize
    details.output[10] = len(sets.trainingSetWordsCounter)  # number of different events in the training set
    details.output[11] = sets.trainingSetWordsCounter[details.inputWord]  # the number of times the event INPUT WORD appears in trainingSet

    # can also be done using probability with lamda=0
    details.output[12] = sets.trainingSetWordsCounter[details.inputWord] / sets.trainingSetSize # MLE based on the training set-no smoothing c(x)\|S|

    # MLE assigns to unseen events if the word ’unseen-word’ is not in the training set
    details.output[13] = sets.trainingSetWordsCounter["unseen-word"] / sets.trainingSetSize

    # MLE based on the training set using λ = 0.10
    details.output[14] = model.probability(sets.trainingSetWordsCounter[details.inputWord], 0.10)
    details.output[15] = model.probability(sets.trainingSetWordsCounter["unseen-word"], 0.10)

    details.output[16] = model.perplexityDict[0.01]  # The perplexity on the validation set using λ = 0.01
    details.output[17] = model.perplexityDict[0.1]  # The perplexity on the validation set using λ = 0.10
    details.output[18] = model.perplexityDict[1.0]  # The perplexity on the validation set using λ = 1.00

    details.output[19] = model.best_lamda  # The value of λ that i found to minimize the perplexity on the validation set
    details.output[20] = model.perplexityDict[model.best_lamda]  # The minimized perplexity on the validation set using the best value you found for λ
    return model

def heldoutTraining(details, sets):
    model = HeldoutModel(details, sets)
    details.output[21] = sets.trainingSetSize
    details.output[22] = sets.validationSetSize
    details.output[23] = model.probability(model.sets.trainingSetWordsCounter[details.inputWord])  # P(Event = INPUT WORD) as estimated by your model.
    details.output[24] = model.probability(0) # P(Event = ’unseen-word’) as estimated by your model.

    return model