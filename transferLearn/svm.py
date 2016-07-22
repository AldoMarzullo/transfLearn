# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
import numpy

def train(features, labels):
    n_samples = len(features)
    data = numpy.asarray(features).reshape((n_samples, -1))
    labels = numpy.asarray(labels).ravel()
    
    # Create a classifier: a support vector classifier
    clf = svm.SVC()
    
    # We learn the smaples on the first half of the digits
    clf.fit(data[:n_samples / 2], labels[:n_samples / 2])
    
    # Now predict the value of the digit on the second half:
    expected = labels[n_samples / 2:]
    predicted = clf.predict(data[n_samples / 2:])
    
    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))