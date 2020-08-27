import numpy as np
from sklearn.model_selection  import train_test_split #数据切割

#定于数据载入数据
filepath=r"C:\Users\Tsinghua-yincheng\Desktop\keras-Day3\pima-indians-diabetes.csv"
dataset=np.loadtxt(filepath,encoding="utf-8",delimiter=',')
X=dataset[:,0:8]
y=dataset[:,8]
#设定随机数种子
np.random.seed(17)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7)#随机打乱数据顺序

from keras.models import Sequential #模型
from keras.layers import Dense #层
model=Sequential()
#model.add(Embedding(input_dim=3,output_dim=3))
model.add(Dense(12,input_dim=8,activation="relu"))
model.add(Dense(6,activation="relu"))
model.add(Dense(1,activation="sigmoid")) #输出

#1-10 5 ，6
#编译模型
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["binary_accuracy"])
#训练模型
model.fit(X,y,epochs=100,validation_split=0.1,batch_size=10,validation_data=(X_test,y_test))
#测试结果
scores=model.evaluate(X,y)
print("测试结果%s :  %f%%"%(model.metrics_names[1],scores[1]*100))

