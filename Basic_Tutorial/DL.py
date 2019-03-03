import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow import keras
def read_data(file):
    x = pd.read_csv("Data/Train.csv",header=None)
    y = pd.read_csv("Data/TrainLabel.csv",header=None)
    y = y.values.ravel()
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)

    return x,Y

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels] = 1
    return one_hot_encode

x,y = read_data(1)
#print(type(y)) #<class 'numpy.ndarray'>
#print(type(x)) #<class 'pandas.core.frame.DataFrame'>
X = x.values
b = pd.read_csv("Data/TrainLabel.csv",header=None)
b = b.values.ravel()
print(b.shape)
#print(X.shape) #(125973, 18)
#print(y.shape) #(125973, 2)
#print(y) encoded 

def Layers():
    model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(18,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
