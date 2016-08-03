# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
from  sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn import cross_validation
import numpy
import input_reader

import inceptionv3 as net
#import myalexnet_forward as net
from sklearn.externals import joblib
from PIL import Image
    
def train(features, labels):
    n_samples = len(features)
    n_test = 100
    data = numpy.asarray(features).reshape((n_samples, -1))
    labels = numpy.asarray(labels).ravel()
    
    # Create a classifier: a support vector classifier
    clf = svm.LinearSVC()
    #clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_features = 50)
    #clf = LDA()
    #clf = AdaBoostClassifier()
    
    # We learn the smaples on the first half of the digits
    print "training svm"
    clf.fit(data[:n_samples - n_test], labels[:n_samples - n_test])
    
    #prediction on test set
    #pred = clf.predict(data[:n_samples - n_test])
    #print("Classification report for classifier %s:\n%s\n"
    #  % (clf, metrics.classification_report(labels[:n_samples - n_test], pred)))
    
    # Now predict the value of the digit on the second half:
    expected = labels[n_samples - n_test:]
    predicted = clf.predict(data[n_samples - n_test:])

    
    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
    
    #stores the trained classifier
    joblib.dump(clf, 'models/clf.pkl', compress=9)

def predict(features):
    print "starting prediction"
    clf = joblib.load('models/clf.pkl')
    
    n_samples = len(features)
    data = numpy.asarray(features).reshape((n_samples, -1))
    return clf.predict(data)
    
    #print("Classification report for classifier on test set %s:\n%s\n"
    #% (clf, metrics.classification_report(expected, predicted)))
    

#model definition  
input_reader.resize = False


#drive = input_reader.create_dataset()
#features, labels = net.extract_features(drive)

#saving features
#numpy.save(input_reader.STORE_FEATURE_PATH, features)
#numpy.save(input_reader.STORE_LABEL_PATH, labels)


features = numpy.load(input_reader.STORE_FEATURE_PATH)
labels = numpy.load(input_reader.STORE_LABEL_PATH)
train(features, labels)


filename = './DRIVE/training/images/21_training.tif'
labelname = './DRIVE/training/1st_manual/21_manual1.gif'

test = input_reader.prepare_image(filename,labelname)
estf, testl = net.extract_features(test)

umpy.save(input_reader.STORE_TEST_PATH, testf)
numpy.save(input_reader.STORE_TEST_PATH_LABEL, testl)

testf = numpy.load(input_reader.STORE_TEST_PATH)
testl = numpy.load(input_reader.STORE_TEST_PATH_LABEL)

size = Image.open(filename).size

prediction = predict(testf)

input_reader.save_as_image(prediction,testl, size)
