import numpy as np
from keras.datasets import cifar10
from keras.layers import *
from keras import backend
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.utils import np_utils
import matplotlib.pyplot as plt
import tensorflow as tf

backend.set_image_data_format('channels_last')


#设定随机数种子
SEED=9
np.random.seed(SEED)

#导入数据
(X_train,y_train),(X_validation,y_validation)=cifar10.load_data()

#将数据格式化到0-1
X_train=X_train.astype('float32')
X_validation=X_validation.astype('float32')
X_train=X_train/255.
X_validation=X_validation/255.

print("X_validation[0].shape = ",X_validation[0].shape)
ep = tf.expand_dims(X_validation[0],0)
print("ep.shape =",ep.shape)

#进行one-hot编码
y_train=np_utils.to_categorical(y_train)
y_validation=np_utils.to_categorical(y_validation)
num_classes=y_train.shape[1]


def CreateCNN(epochs=25):
    model = Sequential()
    # 处理三元色，则strides=（3，3）
    model.add( Conv2D(32, (3, 3), input_shape=(32, 32,3), activation="relu", padding="same", kernel_constraint=maxnorm(3)))  # 卷积层
    model.add(Dropout(rate=0.2))  # 学习率下降
    model.add(Conv2D(32, (3, 3), activation="relu",padding="same",kernel_constraint=maxnorm(3)))  # 中间卷积层
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 池化层
    model.add(Flatten())  # 输入展平。不影响批量大小。  #kernel_constraint=maxnorm(3))核心约束，3原色
    model.add(Dense(units=512, activation="relu",kernel_constraint=maxnorm(3)))  # 中间层
    model.add(Dropout(rate=0.5))  # 学习率下降
    model.add(Dense(units=10, activation="softmax"))  # 输出层
    lrate=0.01
    decay=lrate/epochs
    adam=Adam(lr=lrate,decay=decay)#自定义优化函数
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
    return model
model=CreateCNN()


trained_model = model.fit(X_train,y_train, validation_split=0.2,epochs=25,batch_size=32,verbose=1)



plt.plot(trained_model.history["loss"])
plt.plot(trained_model.history["val_loss"])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","validation"],loc="upper left")
plt.show()

# #accuracy
plt.plot(trained_model.history["accuracy"])
plt.plot(trained_model.history["val_accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train","validation"],loc="upper left")
plt.show()

# model.save("image.h5")
# print(X_validation[0:10].shape)
#
# saved_model = load_model("image.h5")
# print(saved_model.predict(ep))
# y_predict = np.argmax(saved_model.predict(ep),axis=-1)
# print(y_predict)

# score=model.evaluate(x=X_validation,y=y_validation,verbose=0)
# print("图像识别率 %f"%( score[1]))
