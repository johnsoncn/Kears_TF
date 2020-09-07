#构造模型
  #损失函数
  #优化函数
  #神经网络类型
#导入数据训练
#保存模型
#读取模型
#识别


#载入数据
import  numpy  as  np
import  matplotlib.pyplot as plt
from keras.datasets import mnist
mnist1 =mnist.load_data(r"C:\Users\Tsinghua-yincheng\Desktop\Keras-Day43\mnist.npz")
(train_image,train_text),(test_image,test_text)=mnist1
train_image=train_image.reshape((60000,28*28))
train_image=train_image.astype('float32')/255 #RGB 0- 255

test_image=test_image.reshape((10000,28*28))
test_image=test_image.astype('float32')/255 #RGB 0- 255

from  keras.utils import to_categorical
train_text=to_categorical(train_text)
test_text=to_categorical(test_text) #转换类型

#定义模型
from   keras.models import Sequential  #模型。导入顺序模型
from keras.layers import Dense,Dropout #层数
from keras import optimizers   #优化函数
from keras import layers


#基础MLP模型
model = Sequential()
model.add(layers.Dense(512,activation="relu",input_shape=(28*28,)))
model.add(Dropout(rate=0.011))
model.add(layers.Dense(10,activation="softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])


model.fit(train_image,train_text,epochs=5,batch_size=128) #训练
test_loss,test_acc=model.evaluate(test_image,test_text)
print("test_loss",test_loss)
print("识别率",test_acc)


result=model.predict(test_image,batch_size=40,verbose=1)
#0.9 0.001 0.002
result_max=np.argmax(result,axis=1)
test_max=np.argmax(test_text,axis=1)
result_bool=np.equal(result_max,test_max)
true_num=np.sum(result_bool)
print("自测识别率 %f" % (true_num/len(result_bool)))








