import numpy as np
import pandas as pd
from tensorflow import keras
from keras.utils import to_categorical
import datetime
import time
from keras.models import model_from_json
from keras.utils import plot_model
from Evaluate import *
from sklearn.model_selection import train_test_split

def readdt(trainfile,labelfile):
    train = pd.read_csv(trainfile, header=None)
    print(train.shape)
    labels = pd.read_csv(labelfile, header=None)
    labels = labels.values.ravel()
    print(labels)
    #one hot encode labels
    #labels_encoded = to_categorical(labels)
    ##print(labels_encoded)
    #print(train.shape) #(125973, 18)  125973 dòng dữ liệu, 18 thuộc tính
    #print(labels_encoded.shape) #(125973, 2) 2 tương ứng 0 1 binary classification
    return train,labels

#Keras models are trained on Numpy arrays of input data and labels.

def Creat_Model(train,labels):
    #input_dim là số thuộc tính
    input_dim = train.shape[1]
    #Creat_model
    Model = keras.Sequential([
        #Inputlayer
        keras.layers.Dense(512, input_dim= input_dim,activation='relu'),
        keras.layers.Dense(256,activation='relu'),
        keras.layers.Dense(256,activation='relu'),
        keras.layers.Dense(1,activation='sigmoid')
    #chọn hidden layer 10 neurons
    ])
    #Before training a model, need to configure the learning process, which is done via the compile
    Model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    #Training
    print('begin training')
    start = time.time()
    print(datetime.datetime.utcnow())

    Model.fit(train,labels,epochs=10,batch_size=32)

    print('training done at')
    end = time.time()
    print(end - start)
    print('save model to JSON file')
    # serialize model to JSON
    model_json = Model.to_json()
    with open ("Neural_Networks/model.json","w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    Model.save_weights("Neural_Networks/model.h5")
    print("Saved model to disk")
    Model.summary()
    # Output network visualization
    #plot_model(Model,to_file="Neural_Networks/model.png")

def TestModel():
    #load json và tạo model
    json_file = open('Neural_Networks/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    load_model = model_from_json(loaded_model_json)

    #load weights vào model
    load_model.load_weights("Neural_Networks/model.h5")
    #load test data
    test = pd.read_csv("Data/Test.csv", header=None)
    labels_test = load_model.predict(test,batch_size = 32)
    #print(labels_test)
    labels = Threshold(labels_test)
    #print(labels)
    pd.DataFrame(labels_test).to_csv("Neural_Networks/Result.csv", header=None, index=None)
    pd.DataFrame(labels).to_csv("Neural_Networks/Result_with_threshold.csv",header=None,index=None)
    return labels

def Threshold(labels_test):
    labels = []
    for i in labels_test:
        if (i >= 0.25):
            a = 1
            labels.append(a)
        else:
            a = 0
            labels.append(a)
    labels = np.asarray(labels)
    #print(type(labels))
    return labels

#train,labels = readdt('Data/Train.csv','Data/TrainLabel.csv')
#Creat_Model(train,labels)

labels = TestModel()
a = 'Data/TestLabel.csv'

b = 'Neural_Networks/Result_with_threshold.csv'
#Evaluate_Model(b,a)

c = 'SVM/SVMtest.csv'
#print("SVM:")
#Evaluate_Model(c,a)
x1,x2 = train_test_split(labels,test_size = 0.2)
#(len(x1))
#print(x1)
#print(len(x2))
#print(x2)