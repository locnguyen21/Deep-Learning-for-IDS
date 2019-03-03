from sklearn import tree
import pickle
import datetime
import time
import numpy as np

import pandas as pd

def DTrees():
    data = pd.read_csv("train.csv",header=None)
    label = pd.read_csv("trainLabel.csv",header=None)
    print ('done dataload')
    start = time.time()
    print (datetime.datetime.utcnow())
    print ('begin training')
    clf = tree.DecisionTreeClassifier()
    clf.fit(data,label.values.ravel())
    print('training done at')
    end = time.time()
    print(end-start)
    model = 'DTrees/DTrees_model.sav'
    pickle.dump(clf,open(model,'wb'))
    print('save model done')

def TestTrees():
    clf = pickle.load(open('DTrees/DTrees_model.sav', 'rb'))
    testData = pd.read_csv('test.csv', header=None)
    result = clf.predict(testData)
    print(type(result))
    pd.DataFrame(result).to_csv("DTrees/DTreestest.csv", header=None, index=None)

#DTrees()
TestTrees()