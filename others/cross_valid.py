# %%

import numpy as np
from keras.models import Sequential  # 模型
from keras.layers import Dense  # 层
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split  # 数据切割
from keras.wrappers.scikit_learn import KerasClassifier #分类
from sklearn.model_selection import cross_val_score #交叉验证

# 定于数据载入数据
filepath = r"pima-indians-diabetes.csv"
dataset = np.loadtxt(filepath, encoding="utf-8", delimiter=',')
X = dataset[:, 0:8]
y = dataset[:, 8]
# 设定随机数种子
np.random.seed(17)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)  # 随机打乱数据顺序

def build_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))  # 输出
    # 1-10 5 ，6
    # 编译模型
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    return model

model=KerasClassifier(build_fn=build_model,epochs=150,batch_size=10,verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
result=cross_val_score(model,X,y,cv=kfold)
print(result.mean())



