from sklearn import svm
import numpy

def train(features, labels):
    n_samples = len(features)
    data = numpy.asarray(features).reshape((n_samples, -1))

    clf = svm.SVC()
    clf.fit(data, numpy.asarray(labels).ravel())