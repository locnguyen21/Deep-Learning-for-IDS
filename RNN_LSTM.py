from tensorflow import keras
import numpy as np
from keras.models import model_from_json
import datetime
import time
from Evaluate import *
from Logicstic_Regression_Neural_Networks import readdt,Threshold
def Create_RNN(train,labels):

    # Inputlayer
    # LSTM input layer must be 3D
    # The meaning of the 3 input dimensions are: samples, time steps, and features.
    # The number of samples is assumed to be 1 or more.
    # reshape() function takes a tuple as an argument that defines the new shape.
    # number_of_rows_to_process_each_loop, the_time_interval_for_next_move(e.g. per day, per month), column
    train = train.values
    sample = train.shape[0]
    features = train.shape[1]
    #convert 2D to 3D for input RNN
    new_train = np.reshape(train,(sample,features,1)) #shape  = (125973, 18, 1)

    Model = keras.Sequential([

        keras.layers.LSTM(80,input_shape=(features,new_train.shape[2]),
                          activation='tanh',recurrent_activation='hard_sigmoid'),
        keras.layers.Dense(1,activation="tanh")
    # chọn hidden layer 10 neurons
    ])
    Model.compile(optimizer='rmsprop',loss='mse', metrics=['accuracy'])

    #Training
    print('begin training')
    start = time.time()
    print(datetime.datetime.utcnow())

    Model.fit(new_train, labels, epochs=100, batch_size= 32)
    print('training done at')
    end = time.time()
    print(end - start)
    print('save model to JSON file')
    # serialize model to JSON
    model_json = Model.to_json()
    with open("Neural_Networks/RNN_model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    Model.save_weights("Neural_Networks/RNN_model.h5")
    print("Saved model to disk")
    Model.summary()

def TestRNN_Model():
    #load json và tạo model
    json_file = open('Neural_Networks/RNN_model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    load_model = model_from_json(loaded_model_json)

    #load weights vào model
    load_model.load_weights("Neural_Networks/RNN_model.h5")
    #load test data
    test = pd.read_csv("Data/Test.csv", header=None)
    test = test.values
    test = np.reshape(test,(test.shape[0],test.shape[1],1))
    labels_test = load_model.predict(test,batch_size = 32)
    #print(labels_test)
    labels = Threshold(labels_test)
    #print(labels)
    pd.DataFrame(labels_test).to_csv("Neural_Networks/RNN_Result.csv", header=None, index=None)
    pd.DataFrame(labels).to_csv("Neural_Networks/RNN_Result_with_threshold.csv",header=None,index=None)
    return labels

#train,labels = readdt('Data/Train.csv','Data/TrainLabel.csv')
#Create_RNN(train,labels)
#TestRNN_Model()
a = 'Data/TestLabel.csv'

print("RNN-LSTM with 100 epochs:")
b = 'Neural_Networks/RNN_Result_with_threshold.csv'
Evaluate_Model(b,a)