import pandas as pd

def Evaluate_Model(machineresult,testresult):
    machine = pd.read_csv(machineresult,header=None)
    test = pd.read_csv(testresult,header=None)
    #change pandas Dataframe to numpy array tien cho de so sanh
    machine = machine.values
    test = test.values
    a = 0

    for i in range (len(machine)):
        if (machine[i] == test[i]):
            a = a + 1;

    truepositive = 0
    truenagative = 0
    falsepositive = 0
    falsenagative = 0

    for i in range (len(machine)):
        #tinh true positive = 0
        if (machine[i] == test[i] and test[i] == 0):
            truepositive = truepositive + 1
        #tinh true negative = 1
        elif (machine[i] == test[i] and test[i] == 1):
            truenagative = truenagative + 1
        #tinh false positive
        elif (machine[i] != test[i] and test[i] == 1):
            falsepositive = falsepositive + 1
        else:
            falsenagative = falsepositive + 1

    precision = truepositive * 100 / (truepositive + falsepositive)
    recall = truepositive * 100 / (truepositive + falsenagative)

    print('Accuracy: ' + str(a * 100 / len(machine)))
    print('Precision : ' + str(precision))
    print('Recall: ' + str(recall))

a = 'testLabel.csv'
b = 'KNN/KNNtest.csv'
c = 'SVM/SVMtest.csv'
d = 'DTrees/DTreestest.csv'
e = 'KNN/KNN_new_test.csv'
g = 'KNN/KNN_train_acc.csv'
h = 'trainLabel.csv'
i = 'KNN/KNN_train_acc1.csv'

