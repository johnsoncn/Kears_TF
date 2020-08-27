from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils

import matplotlib.pyplot as plt
#设定随机数种子
SEED=9
np.random.seed(SEED)
(X_train,y_train),(X_validation,y_validation)=cifar10.load_data()

print(X_train[:100])
y_train=np_utils.to_categorical(y_train)
# print(y_train)

# print(X_train.shape,y_train.shape)
# #(50000, 32, 32, 3)
# #(50000, 1)
#
plt.figure(1)
plt.imshow(X_validation[3])
plt.title(y_train[2])
plt.show()
# print(X_train.shape,X_validation.shape)
# #将数据格式化到0-1,  0-255 RGB图片可以由三个数描述(123,123,213)，即三元色
# X_train=X_train.astype('float32')
# X_validation=X_validation.astype('float32')
# print(X_validation[0][0])
# X_train=X_train/255.
# X_validation=X_validation/255.
#
# print(X_validation[0][0])
#
# print(X_train.shape,X_validation.shape)


