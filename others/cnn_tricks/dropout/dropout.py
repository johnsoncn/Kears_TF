import numpy as np
from keras.models import Sequential  # 模型
from keras.layers import Dense  # 层
from keras.layers import Dropout  #
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split  # 数据切割
from keras.wrappers.scikit_learn import KerasClassifier  # 分类
from sklearn.model_selection import cross_val_score  # 交叉验证
from sklearn import datasets
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.model_selection import KFold

dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

print(X, y)
seed = 17
np.random.seed(17)


def build_model(init="glorot_uniform"):
    model = Sequential()
    model.add(Dropout(rate=0.2, input_shape=(4,)))
    model.add(Dense(units=4, activation="relu", kernel_initializer=init))

    model.add(Dropout(rate=0.2))
    model.add(Dense(units=6, activation="relu", kernel_initializer=init))

    model.add(Dropout(rate=0.2))
    model.add(Dense(3, activation="softmax", kernel_initializer=init))

    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model


model = KerasClassifier(build_fn=build_model, epochs=500, batch_size=10, verbose=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
result = cross_val_score(model, X, y, cv=kfold)
print(result.mean())
print(result.std())


def build_modelwithoutdropout(init="glorot_uniform"):
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation="relu", kernel_initializer=init))
    model.add(Dense(6, activation="relu", kernel_initializer=init))
    model.add(Dense(3, activation="softmax", kernel_initializer=init))

    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model


model = KerasClassifier(build_fn=build_modelwithoutdropout, epochs=500, batch_size=10, verbose=0)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
result = cross_val_score(model, X, y, cv=kfold)
print(result.mean())
print(result.std())
