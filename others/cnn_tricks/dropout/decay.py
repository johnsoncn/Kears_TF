
import numpy as np
from keras.models import Sequential  # 模型
from keras.layers import Dense  # 层
from keras.layers import Dropout #
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split  # 数据切割
from keras.wrappers.scikit_learn import KerasClassifier #分类
from sklearn.model_selection import cross_val_score #交叉验证
from  sklearn import datasets
from keras.utils  import to_categorical
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from math  import pow,floor
from keras.callbacks import LearningRateScheduler
#导入数据
dataset=datasets.load_iris() #载入分类数据
X=dataset.data
y=dataset.target



print(X,y)
seed=17
np.random.seed(17)

def step_decay(epoch):
    init_lrate=0.3
    drop=0.1
    epochs_drop=1 #每10个衰减0.5
    lrate=init_lrate*pow(drop,floor(1+epoch)/epochs_drop) #指数衰减
    return  lrate



def build_model(init="glorot_uniform"):
    model = Sequential()
    model.add(Dropout(rate=0.2,input_shape=(4,)))
    model.add(Dense(units=4, activation="relu",kernel_initializer=init))

    model.add(Dropout(rate=0.2))
    model.add(Dense(units=6, activation="relu",kernel_initializer=init))

    model.add(Dropout(rate=0.2))
    model.add(Dense(3, activation="softmax",kernel_initializer=init))  # 输出
    # 1-10 5 ，6
    # 编译模型
    #随机梯度下降
    learning_rate=0.05
    momentum=0.5
    #衰减
    decay_rate=0
    #y=ax+b ,0.005 0.0010 0.0015
    sgd=SGD(lr=learning_rate,momentum=momentum,decay=decay_rate,nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model

lrate= LearningRateScheduler(step_decay)
model=KerasClassifier(build_fn=build_model,epochs=500,
                      batch_size=10,verbose=2,callbacks=[lrate])
model.fit(X,y)
scores=model.score(X,y)
print(scores)