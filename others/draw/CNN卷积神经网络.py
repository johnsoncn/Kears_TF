from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras import backend
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model

backend.set_image_data_format("channels_first")#设置图像格式
#导入数据
(X_train,y_train),(X_validation,y_validation)=mnist.load_data()


#显示四张手写数据集
plt.subplot(221)
plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1],cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2],cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3],cmap=plt.get_cmap('gray'))
plt.show()

seed=189
np.random.seed(seed)

print(X_train.shape)

num_pixels=X_train.shape[1]*X_train.shape[2]

print(num_pixels)

X_train=X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_validation=X_validation.reshape(X_validation.shape[0],1,28,28).astype('float32')

#格式化数据到0-1
X_train=X_train/255
X_validation=X_validation/255

#进行One-hot编码
y_train=np_utils.to_categorical(y_train)
y_validation=np_utils.to_categorical(y_validation)

num_classes=y_validation.shape[1]
print(num_classes)


#基础CNN模型
model = Sequential()
model.add(Conv2D(32,(5,5),input_shape=(1,28,28),activation="relu")) #卷积层
model.add(MaxPooling2D(pool_size=(2,2)))#池化层
model.add(Dropout(rate=0.2))#学习率下降
model.add(Flatten()) #输入展平。不影响批量大小。
model.add(Dense(units=128,activation="relu"))#中间层
model.add(Dense(units=10,activation="softmax"))#输出层
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(X_train,y_train,epochs=1,batch_size=200) #训练
score=model.evaluate(X_validation,y_validation)
model.save("mnist.h5")
print("CNN  %f" %(score[1]))

plot_model(model,to_file="model_1.png",show_shapes=True,show_layer_names=True,rankdir="TB")
plt.figure(figsize=(10,10))
img=plt.imread("model.png")
#plt.show(img)
plt.axis("off")
plt.show()