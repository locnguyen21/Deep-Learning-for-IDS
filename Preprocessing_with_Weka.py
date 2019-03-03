import pandas as pd
import csv
def protocoltoInt(data):
    protocol = {
        "tcp": 1, "udp":2, "icmp":3
    }
    return protocol[data]

def flagtoInt(data):
    flag = {
        "SF": 1, "S0": 2, "REJ": 3, "RSTR": 4, "RSTO": 5,
        "S1": 6, "SH": 7, "S2": 8, "RSTOS0": 9, "S3": 10, "OTH": 11
    }
    return flag[data]

#file csv co san h chi loc thuoc tinh
def dataProcess(file,fileInput,fileOutput):
    Data = pd.read_csv(file)
    #chuyen tu pandas thanh numpy
    Data = Data.values
    print(type(Data))
    inwr = csv.writer(fileInput)
    outw = csv.writer(fileOutput)
    for flow in Data:
        da, la = Preprocessing(flow)
        inwr.writerow(da)
        outw.writerow(la)


def Preprocessing(flow):
    dat = []
    protocol = protocoltoInt(flow[1])
    flag = flagtoInt(flow[3])
    dat = [flow[0],protocol,flag,flow[4],flow[5],flow[7],flow[8],flow[9],flow[10],
    flow[12],flow[14],flow[16],flow[17],flow[18],flow[22],flow[23],flow[31],flow[32]]
    if (flow[-1] == "normal"):
        labl = [0]
    else: labl = [1]
    return dat, labl

trainFile = open('Data/Train.csv','w+',newline='')
trainLabel = open('Data/TrainLabel.csv','w+',newline='')
dataProcess("Data/KDDTrain.csv",trainFile,trainLabel)

testFile = open('Data/Test.csv','w+',newline='')
testLabel = open('Data/TestLabel.csv','w+',newline='')
dataProcess("Data/KDDTest.csv",testFile,testLabel)

