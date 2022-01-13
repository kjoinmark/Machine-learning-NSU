import numpy as np


def mera(x):
    return np.power(((x) ** 2), 1 / 2)


class KnnClassification(object):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.numberOfClasses = 0
        self.classes = None
        self.trainX = None
        self.trainY = None

    def fit(self, trainDataX, trainDataY):
        self.trainX = trainDataX
        self.classes = list(zip(np.unique(trainDataY), np.arange(1, 1 + len(np.unique(trainDataY)), dtype=int)))
        self.numberOfClasses = len(self.classes)
        self.trainY = np.zeros(len(trainDataY), dtype=int)
        for i in range(len(trainDataY)):
            for cl in self.classes:
                if cl[0] == trainDataY[i]:
                    self.trainY[i] = cl[1]
                    break

    def predict(self, testData):
        testLabels = []
        for testObj in testData:
            featureMatrix = self.trainX.copy()
            for i in range(len(self.trainY)):
                featureMatrix[i] = abs(featureMatrix[i] - testObj)
            testDist = list(zip(featureMatrix.sum(axis=1), self.trainY))
            testDist.sort()
            stat = np.zeros(self.numberOfClasses)
            for d in testDist[0:self.n_neighbors]:
                stat[d[1]-1] += 1
            tmp = sorted(zip(stat, range(1, 1 + self.numberOfClasses)), reverse=True)
            testLabels.append(self.classes[tmp[0][1] - 1][0])
        return testLabels
