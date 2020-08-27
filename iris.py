



import numpy as np
from keras.models import Sequential  # 模型
from keras.layers import Dense  # 层
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split  # 数据切割
from keras.wrappers.scikit_learn import KerasClassifier #分类
from sklearn.model_selection import cross_val_score #交叉验证
from  sklearn import datasets
#导入数据
dataset=datasets.load_iris() #载入分类数据
X=dataset.data
y=dataset.target
# print(y)
np.random.seed(17)

# kernel_initializer = glorot_uniform，打碎数据，均匀分布初始化

def build_model(optimeizer="adam",init="glorot_uniform"):
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation="relu",kernel_initializer=init)) # 输入数据为4行
    model.add(Dense(6, activation="relu",kernel_initializer=init))
    model.add(Dense(3, activation="softmax",kernel_initializer=init))  # 输出，结果分为3类
    # 1-10 5 ，6
    # 编译模型
    model.compile(loss="categorical_crossentropy", optimizer=optimeizer, metrics=["accuracy"])
    return model

model=KerasClassifier(build_fn=build_model,epochs=200,batch_size=10,verbose=1)

# print(model)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)
result=cross_val_score(model,X,y,cv=kfold)

# print(result)
print(result.mean())
print(result.std()) # 方差越小模型越好

