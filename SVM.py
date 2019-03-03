import pandas as pd
import numpy as np
from sklearn import svm
import time
import pickle
import datetime

def SVM():
    data = pd.read_csv("train.csv", header=None)
    label = pd.read_csv("trainLabel.csv", header=None)
    print('done dataload')
    start = time.time()
    print(datetime.datetime.utcnow())
    print('begin training')
    clf = svm.SVC(gamma='scale')
    clf.fit(data, label.values.ravel())
    print('training done at')
    end = time.time()
    print(end - start)
    model = 'SVM/SVM_model.sav'
    pickle.dump(clf, open(model, 'wb'))
    print('save model done')

#SVM()

def TestSVM():
    clf = pickle.load(open('SVM/SVM_model.sav', 'rb'))
    testData = pd.read_csv('test.csv', header=None)
    result = clf.predict(testData)
    print(type(result))
    pd.DataFrame(result).to_csv("SVM/SVMtest.csv", header=None, index=None)

TestSVM()