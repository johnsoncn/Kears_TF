
import numpy as np
from keras.models import Sequential  # 模型
from keras.layers import Dense  # 层
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split  # 数据切割
from keras.wrappers.scikit_learn import KerasClassifier #分类
from sklearn.model_selection import cross_val_score #交叉验证
from  sklearn import datasets
from keras.utils  import to_categorical
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

dataset=datasets.load_iris()
X=dataset.data
y=dataset.target

#转化数据编码
Y_labels=to_categorical(y,num_classes=3)

print(X,y)
np.random.seed(17)

def build_model(optimeizer="adam",init="glorot_uniform"):
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation="relu",kernel_initializer=init))
    model.add(Dense(6, activation="relu",kernel_initializer=init))
    model.add(Dense(3, activation="softmax",kernel_initializer=init))

    model.compile(loss="categorical_crossentropy", optimizer=optimeizer, metrics=["accuracy"])
    return model

model=build_model()


#设置检查点，检测
filepath="hdf5/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint=ModelCheckpoint(filepath=filepath,monitor="val_accuracy",mode="max",save_best_only=True)
callback_list=[checkpoint]
model.fit(X,Y_labels,validation_split=0.2,epochs=500,batch_size=5,verbose=1,callbacks=callback_list)#训练


#保存最优结果
filepath="hdf5/weights.best.h5"
checkpoint=ModelCheckpoint(filepath=filepath,monitor="val_accuracy",mode="max",save_best_only=True)
callback_list=[checkpoint]
model.fit(X,Y_labels,validation_split=0.2,epochs=500,batch_size=5,verbose=1,callbacks=callback_list)#训练


