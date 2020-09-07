import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import math
import numpy as np
#构造MLP，多层感知机
SEED=7 #随机数种子
MAX_FEATURES=5000
MAX_WORDS=500
OUTPUT_DIM=2 #正面，负面
BATCH_SIZE=2#批量处理
EPOCHS=200 #训练测试
filepath=r"international-airline-passengers.csv"
footer=3
look_back=1

import sys
np.set_printoptions(threshold=sys.maxsize)

def create_dataset(dataset):
    dataX,dataY=[],[]
    for i in range (len(dataset)-look_back-1):
        x=dataset[i:i+look_back,0] #构造时间线，1天间隔
        dataX.append(x)
        y=dataset[i+look_back,0]
        dataY.append(y)
    return np.array(dataX),np.array(dataY)



if  __name__=="__main__":
    #读取数据
    data=pd.read_csv(filepath,usecols=[1],engine="python",skipfooter=footer)
    # print(data.shape)
    dataset=data.values.astype("float32")
    # print(dataset)#打印数据
    train_size=int(len(dataset)*0.67)  #训练的数据
    validation_size=len(dataset) - train_size #测试的数据
    # print(train_size,validation_size)
    train,validation=dataset[0:train_size,:],dataset[train_size:len(dataset),:]#数据切割
    #
    x_train,y_train=create_dataset(train)
    x_valition, y_valition= create_dataset(validation)  #数据切割
    #
    print(x_train, y_train)
    print(x_train.shape, y_train.shape)

