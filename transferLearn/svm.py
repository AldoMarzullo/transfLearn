from sklearn import svm
import numpy

def train(features, label):
    n_samples = len(features)
    data = features.reshape((n_samples, -1))
    clf = svm.SVC()
    clf.fit(data, label)  