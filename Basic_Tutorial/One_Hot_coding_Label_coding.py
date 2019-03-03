import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

#LabelEncoder
le = preprocessing.LabelEncoder()
le.fit(["loc","tra","vinh","vinh","hoang","loc","loc"])
print(list(le.classes_))
#Encoding lay vi tri trong mang classes
#['hoang', 'loc', 'tra', 'vinh']
a = le.transform(["loc","loc","vinh"])
#vi tri cua loc la 1, vinh la 3
# a = [1,1,3]
#print(a)

#reverse nguoc lai
#b = ['vinh', 'loc', 'tra', 'vinh']
b = list(le.inverse_transform([3,1,2,3]))
#print(b)


#ONE HOT ENCODING
enc = OneHotEncoder(handle_unknown= 'ignore')
X = [['Apple',95], ['Chicken',231], ['Broccli',50],['Apple',40]]
enc.fit(X)
print(enc.categories_)
a = enc.transform(X).toarray()
print(a)