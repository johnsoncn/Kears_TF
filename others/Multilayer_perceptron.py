
import numpy as np
#定于数据载入数据
filepath=r"C:\Users\Tsinghua-yincheng\Desktop\keras-Day3\pima-indians-diabetes.csv"
dataset=np.loadtxt(filepath,encoding="utf-8",delimiter=',')
#print(dataset)
X=dataset[:,0:8]
y=dataset[:,8]
#print(X) #判断的八个特征
#print(y) #判断的结果
#X=np.expand_dims(X,axis=1)
#y=np.expand_dims(y,axis=1)
print(X)
print(y)


from keras.models import Sequential #模型
from keras.layers import Dense #层
from keras.layers import Embedding #内置

#have 3 dimensions, but got array with shape (768, 8)
#创建模型
model=Sequential()
#model.add(Embedding(input_dim=3,output_dim=3))
model.add(Dense(12,input_dim=8,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(8,activation="softmax"))
model.add(Dense(1,activation="sigmoid")) #输出

#1-10 5 ，6
#编译模型
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["binary_accuracy"])
#训练模型
model.fit(X,y,epochs=100,validation_split=0.1)
#测试结果
scores=model.evaluate(X,y)
print("测试结果%s :  %f%%"%(model.metrics_names[1],scores[1]*100))

#测试结果accuracy :  63.104169% 3层
#测试结果accuracy :  65.104169% 4层
#测试结果accuracy :  64.973956%% 4层
#测试结果accuracy :  65.494794% 5层修改了激活函数
#测试结果accuracy :  65.494794% 5层修改了损失函数
#测试结果accuracy :  65.104169%   修改nadam
#测试结果accuracy :  65.364581%   修改adam
#测试结果binary_accuracy :  65.104169%

