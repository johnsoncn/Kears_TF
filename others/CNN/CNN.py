from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras import backend
from keras.layers import Dropout
import sys
np.set_printoptions(threshold=sys.maxsize)

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


backend.set_image_data_format("channels_last") #设置图像格式
#导入数据
(X_train,y_train),(X_test,y_test)=mnist.load_data()

# 计算像素，num_pixels=28*28
num_pixels=X_train.shape[1]*X_train.shape[2]

# #显示四张手写数据集
# plt.subplot(221)
# plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[1],cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[2],cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[3],cmap=plt.get_cmap('gray'))
# plt.show()
#
# seed=189
# np.random.seed(seed)


# 把原测试集（60000，28，28） reshape为 （60000，1，28，28）
X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32') #（10000，1，28，28）
#格式化数据到0-1
X_train=X_train/255
X_test=X_test/255



# 对训练集和验证集的label进行One-hot编码
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]
print(num_classes)

print("X_train.shape = ",X_train.shape)

model = Sequential()
model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.2))#学习率下降
"""
Convolution卷积层之后是无法直接连接Dense全连接层的，需要把Convolution层的数据压平（Flatten），然后就可以直接加Dense层了。
也就是把 (height,width,channel)的数据压缩成长度为 
    height × width × channel 的一维数组，然后再与 FC层连接，这之后就跟普通的神经网络无异了。
"""
model.add(Flatten()) #输入展平。不影响批量大小。
model.add(Dense(units=128,activation="relu"))#中间层
model.add(Dense(units=10,activation="softmax"))#输出层
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])


# trained_model = model.fit(X_train,y_train,validation_split=0.2,epochs=10,batch_size=200,verbose=1)
# # score=model.evaluate(X_validation,y_validation)
# # print("CNN  %f" %(score[1]))
#
# print(trained_model.history.keys())
#
# plt.plot(trained_model.history["loss"])
# plt.plot(trained_model.history["val_loss"])
# plt.title("model loss")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.legend(["train","validation"],loc="upper left")
# plt.show()
#
# model.save("MLP.h5")
model.summary()

############################ 加载模型进行预测 ##################################
#
# from keras import backend as K
#
# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
#
# saved_model = load_model("MLP.h5")
# y_predict = np.argmax(saved_model.predict(X_test), axis=-1)
# # 由于y_test.shape = (10000,10)的one_hot向量，那么计算recall时也需要转为one_hot向量
# y_predict = np_utils.to_categorical(y_predict)
#
# print(y_test.shape)
# print(y_predict.shape)
# print(y_predict)
#
# re = recall(y_test,y_predict)
# print("recall = {}".format(re))

