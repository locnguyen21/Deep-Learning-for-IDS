import numpy as np
import csv
#doc du lieu file nay cho vao array file khac
def dataPre(filename,fileInput,fileOutput):
    with open(filename) as file:
        contentFile = file.readlines()
        print(type(contentFile))
        inwr = csv.writer(fileInput)
        outw = csv.writer(fileOutput)
        for flow in contentFile:
            data, label = PreprocessingLineTest(flow)
            #data, label = PreprocessingLineTrain(flow)
            inwr.writerow(data)
            outw.writerow(label)

#NSL-KDD co 3 loai du lieu: normal, numeric va binary
#xu li du lieu normal
def ProtocoltoInt(data):
    protocol = {
        "tcp": 1, "udp":2, "icmp":3
    }
    return protocol[data]

def FlagtoInt(data):
    flag = {
        "SF": 1, "S0": 2, "REJ": 3, "RSTR": 4, "RSTO": 5,
        "S1": 6, "SH": 7, "S2": 8, "RSTOS0": 9, "S3": 10, "OTH": 11
    }
    return flag[data]

def Scallingdata(val,max):
    return val/max

#data se bao gom 19 thuoc tinh
#lay du lieu data, tien xu ly
def PreprocessingLineTrain(line):
    data = []
    a = line.split(",")
    duration= Scallingdata(int(a[0]),42908)
    protocol = ProtocoltoInt(a[1])
    flag = FlagtoInt(a[3])
    src_bytes = Scallingdata(int(a[4]),1379963888)
    dst_bytes = Scallingdata(int(a[5]),1309937401)
    wrong_fragment = Scallingdata(int(a[7]),3)
    urgent = Scallingdata(int(a[8]),3)
    hot = Scallingdata(int(a[9]),77)
    num_failed_login = Scallingdata(int(a[10]),5)
    num_compromised = Scallingdata(int(a[12]),7479)
    su_attempted = Scallingdata(int(a[14]),2)
    num_root = Scallingdata(int(a[15]),7468)
    num_file_creations= Scallingdata(int(a[16]),43)
    num_shell = Scallingdata(int(a[17]),2)
    num_access_files = Scallingdata(int(a[18]),9)
    count = Scallingdata(int(a[22]),511)
    srv_count = Scallingdata(int(a[23]),511)
    dst_host_count = Scallingdata(int(a[31]),255)
    dst_host_srv_count = Scallingdata(int(a[32]),255)
    data = [duration,protocol,flag,src_bytes,dst_bytes,wrong_fragment,urgent,hot,num_failed_login,num_compromised,
                su_attempted, num_root, num_file_creations, num_shell, num_access_files,count,
                srv_count,dst_host_count,dst_host_srv_count    ]
    if (a[-2] == "normal"):
        label = [0]
    else:
        label = [1]
    return data, label

# lay du lieu nhan label
# def BinaryLabel(contenfile):
#     y = []
#     for line in contentfile:
#         a = line.split(",")
#         if (a[-2] == "normal"):
#             label = 0
#         else: label = 1
#         y.append(label)
#     return y

#them newline ='' de khong bi blank line trong file csv
def PreprocessingLineTest(line):
    data = []
    a = line.split(",")
    duration = Scallingdata(int(a[0]), 57715)
    protocol = ProtocoltoInt(a[1])
    flag = FlagtoInt(a[3])
    src_bytes = Scallingdata(int(a[4]), 62825648)
    dst_bytes = Scallingdata(int(a[5]), 1345927)
    wrong_fragment = Scallingdata(int(a[7]), 3)
    urgent = Scallingdata(int(a[8]), 3)
    hot = Scallingdata(int(a[9]), 101)
    num_failed_login = Scallingdata(int(a[10]), 4)
    num_compromised = Scallingdata(int(a[12]), 796)
    su_attempted = Scallingdata(int(a[14]), 2)
    num_root = Scallingdata(int(a[15]), 878)
    num_file_creations = Scallingdata(int(a[16]), 100)
    num_shell = Scallingdata(int(a[17]), 5)
    num_access_files = Scallingdata(int(a[18]), 4)
    count = Scallingdata(int(a[22]), 511)
    srv_count = Scallingdata(int(a[23]), 511)
    dst_host_count = Scallingdata(int(a[31]), 255)
    dst_host_srv_count = Scallingdata(int(a[32]), 255)
    data = [duration, protocol, flag, src_bytes, dst_bytes, wrong_fragment, urgent, hot, num_failed_login,
            num_compromised,
            su_attempted, num_root, num_file_creations, num_shell, num_access_files, count,
            srv_count, dst_host_count, dst_host_srv_count]
    if (a[-2] == "normal"):
        label = [0]
    else:
        label = [1]
    return data, label


#trainFile = open('train.csv','w+',newline='')
#trainLabel = open('trainLabel.csv','w+',newline='')
#contentfile = dataPre("KDDTrain+.txt",trainFile,trainLabel)

testFile = open('test.csv','w+',newline='')
testLabel = open('testLabel.csv','w+',newline='')
content = dataPre("KDDTest+.txt",testFile,testLabel)
#print (y)

