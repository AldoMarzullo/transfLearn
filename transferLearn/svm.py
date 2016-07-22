# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
import numpy
import input_reader
import myalexnet_forward as net
from sklearn.externals import joblib
from PIL import Image
    
def train(features, labels):
    n_samples = len(features)
    data = numpy.asarray(features).reshape((n_samples, -1))
    labels = numpy.asarray(labels).ravel()
    
    # Create a classifier: a support vector classifier
    clf = svm.SVC()
    
    # We learn the smaples on the first half of the digits
    print "training svm"
    clf.fit(data[:n_samples / 2], labels[:n_samples / 2])
    
    # Now predict the value of the digit on the second half:
    expected = labels[n_samples / 2:]
    predicted = clf.predict(data[n_samples / 2:])
    

    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
    
    #stores the trained classifier
    joblib.dump(clf, 'models/clf.pkl', compress=9)

def predict(features):
    clf = joblib.load('models/clf.pkl')
    return clf.predict(features)
    
    #print("Classification report for classifier on test set %s:\n%s\n"
    #% (clf, metrics.classification_report(expected, predicted)))
    
    
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
testf, testl = net.extract_features(test)

size = Image.open(filename).size
prediction = predict(testf)

print (numpy.shape(prediction))
input_reader.save_as_image(prediction, size)