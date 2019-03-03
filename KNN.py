from sklearn import neighbors
import pickle
import datetime
import time
import numpy as np
import csv
import pandas as pd
def KNN():
    data = pd.read_csv("train.csv",header=None)
    label = pd.read_csv("trainLabel.csv",header=None)
    print ('done dataload')
    start = time.time()
    print (datetime.datetime.utcnow())
    print ('begin training')
    clf = neighbors.KNeighborsClassifier()
    clf.fit(data,label.values.ravel())
    print('training done at')
    end = time.time()
    print(end-start)
    model = 'KNN/KNN_model.sav'
    pickle.dump(clf,open(model,'wb'))
    print('save model done')

def TestKNN():
    clf = pickle.load(open('KNN/KNN_model.sav','rb'))
    testData = pd.read_csv('train.csv',header=None)
    result = clf.predict(testData)
    print((result))
    pd.DataFrame(result).to_csv("KNN/KNN_train_acc1.csv",header=None,index=None)

def KNNwith():
    dulieu = pd.read_csv("Data/Train.csv",header=None)
    nhan = pd.read_csv("Data/TrainLabel.csv",header=None)
    print('done dataload')
    batdau = time.time()
    print(datetime.datetime.utcnow())
    print('begin training')
    cl = neighbors.KNeighborsClassifier()
    cl.fit(dulieu,nhan.values.ravel())
    print('training done at')
    ketthuc = time.time()
    print (ketthuc- batdau)
    mohinh = 'KNN/KNN_new_model.sav'
    pickle.dump(cl, open(mohinh, 'wb'))
    print('save model done')

#KNNwith()
TestKNN()