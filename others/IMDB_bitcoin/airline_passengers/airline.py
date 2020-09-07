import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import math
import numpy as np

"""
MLP时间线预测：滑动窗口为1，即根据昨天预测今天

把上一个数据作为输入，下一个作为输出（label），以此循环，并分别保存为x_train和y_train

然后喂到模型里，学习到上下关系
"""



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

def create_dataset(dataset):
    dataX,dataY=[],[]
    for i in range (len(dataset)-look_back-1):
        x=dataset[i:i+look_back,0] #构造时间线，1天间隔
        dataX.append(x)
        y=dataset[i+look_back,0]
        dataY.append(y)
    return np.array(dataX),np.array(dataY)

#构造模型
def build_model():
    model=Sequential()
    model.add(Dense(units=8,input_dim=look_back,activation="relu"))
    model.add(Dense(units=1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    return model

if  __name__=="__main__":
    #读取数据
    data=pd.read_csv(filepath,usecols=[1],engine="python",skipfooter=footer)
    #print(data)
    dataset=data.values.astype("float32")
    print(dataset)#打印数据
    train_size=int(len(dataset)*0.67)  #训练的数据
    validation_size=len(dataset)- train_size #测试的数据
    train,validation=dataset[0:train_size,:],dataset[train_size:len(dataset),:]#数据切割

    x_train,y_train=create_dataset(train)
    x_valition, y_valition= create_dataset(validation)  #数据切割


    #训练模型
    model=build_model()
    model.fit(x_train,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1)
    #评估模型
    train_score=model.evaluate(x_train,y_train,verbose=0)
    print("%f" % train_score)
    print("%f" %  math.sqrt(train_score))
    #预测数据
    predict_train=model.predict(x_train,batch_size=16)
    predict_valition=model.predict(x_valition,batch_size=16)

    # print(predict_train)
    #训练绘制
    predict_train_plot=np.empty_like(dataset)
    predict_train_plot[:,:]=np.nan #填充
    predict_train_plot[look_back:len(predict_train)+look_back,:]=predict_train
    #训练绘制
    predict_valition_plot=np.empty_like(dataset)
    predict_valition_plot[:,:]=np.nan #填充
    predict_valition_plot[len(predict_train)+look_back*2+1:len(dataset)-1,:]=predict_valition

    plt.plot(dataset,color="red")
    plt.plot(predict_train_plot,color="blue")
    plt.plot(predict_valition_plot,color="black")
    plt.show()
