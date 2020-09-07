import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import  KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
import tensorflow as tf

file=r'bank.csv'
dataset=pd.read_csv(file,delimiter=';')
# print(dataset.head())
# print(dataset.info())
# print(dataset['job'].unique())

dataset['job'] = dataset['job'].replace(to_replace=['admin.', 'unknown', 'unemployed', 'management',
                                                    'housemaid', 'entrepreneur', 'student', 'blue-collar',
                                                    'self-employed', 'retired', 'technician', 'services'],
                                        value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
dataset['marital'] = dataset['marital'].replace(to_replace=['married', 'single', 'divorced'], value=[0, 1, 2])
dataset['education'] = dataset['education'].replace(to_replace=['unknown', 'secondary', 'primary', 'tertiary'],
                                                    value=[0, 2, 1, 3])
dataset['default'] = dataset['default'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['housing'] = dataset['housing'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['loan'] = dataset['loan'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['contact'] = dataset['contact'].replace(to_replace=['cellular', 'unknown', 'telephone'], value=[0, 1, 2])
dataset['poutcome'] = dataset['poutcome'].replace(to_replace=['unknown', 'other', 'success', 'failure'],
                                                  value=[0, 1, 2, 3])
dataset['month'] = dataset['month'].replace(to_replace=['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
dataset['y'] = dataset['y'].replace(to_replace=['no', 'yes'], value=[0, 1])
array=dataset.values
at = tf.convert_to_tensor(array,dtype=tf.int32)
# print(at)
x=array[:,0:16]
Y=array[:,16]
# print(x)
# print(Y)

# 设置随机种子
seed = 7
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.33, random_state=seed)

import sys
np.set_printoptions(threshold=sys.maxsize)



print("X_train={}, X_test={}, y_train={}, y_test={}".format(X_train, X_test, y_train, y_test ))




def build_model():
    model = Sequential()
    model.add(Dense(16,activation='relu',input_shape=(16,)))  # input arrays of shape (*,16) and output (*,32)
    # 在第一层之后，你就不再需要指定输入的尺寸了
    model.add(Dense(16,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',# 还可以通过optimizer = optimizers.RMSprop(lr=0.001)来为优化器指定参数
                  loss='binary_crossentropy', # 等价于loss = losses.binary_crossentropy
                  metrics=['accuracy']) # 等价于metrics = [metircs.binary_accuracy]

    return model

model = build_model()
model.summary()

# model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=50, batch_size=10)
# model.save("bank_validation.h5")

from sklearn.metrics import f1_score, precision_score, recall_score

loaded_model = load_model('bank_validation.h5')

# y_predict = loaded_model.predict(X_test)
y_predict = np.argmax(loaded_model.predict(x), axis=-1)
print(type(y_predict))
print(y_predict)