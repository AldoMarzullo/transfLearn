# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
from  sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn import cross_validation
import numpy
import input_reader

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

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
    
    #clf = SVC(kernel='rbf', C=8, gamma=0.5)
    #clf = SVC(kernel='rbf', C=16, gamma=0.5)
    clf = svm.LinearSVC(C=0.5,loss='squared_hinge')
    
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
    
    """C = []
    for i in range(-5,15):
      C.append(pow(2,i))
	       
    g = []
    for i in range(-15,3):
      g.append(pow(2,i))
	       
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': g,'C': C},
			{'kernel': ['linear'], 'C': C}]


    #'kernel': 'rbf', 'C': 8, 'gamma': 0.5
    #'kernel': 'rbf', 'C': 16, 'gamma': 0.5
    print("# Tuning hyper-parameters")
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10, scoring='f1_macro', n_jobs=-1, verbose=2)
    clf.fit(data[:n_samples - n_test], labels[:n_samples - n_test])

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
	print("%0.3f (+/-%0.03f) for %r"
	      % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = expected, clf.predict(data[n_samples - n_test:])
    print(classification_report(y_true, y_pred))
    print()"""


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


drive = input_reader.create_dataset()
features, labels = net.extract_features(drive)

#saving features
numpy.save(input_reader.STORE_FEATURE_PATH, features)
numpy.save(input_reader.STORE_LABEL_PATH, labels)


features = numpy.load(input_reader.STORE_FEATURE_PATH)
labels = numpy.load(input_reader.STORE_LABEL_PATH)
train(features, labels)


filename = './DRIVE/training/images/21_training.tif'
labelname = './DRIVE/training/1st_manual/21_manual1.gif'

test = input_reader.prepare_image(filename,labelname)
testf, testl = net.extract_features(test)

numpy.save(input_reader.STORE_TEST_PATH, testf)
numpy.save(input_reader.STORE_TEST_PATH_LABEL, testl)

testf = numpy.load(input_reader.STORE_TEST_PATH)
testl = numpy.load(input_reader.STORE_TEST_PATH_LABEL)

size = Image.open(filename).size

prediction = predict(testf)

input_reader.save_as_image(prediction,testl, size)
