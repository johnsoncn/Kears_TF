import numpy as np
from keras.models import Sequential  # 模型
from keras.layers import Dense  # 层
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split  # 数据切割
from keras.wrappers.scikit_learn import KerasClassifier #分类
from sklearn.model_selection import cross_val_score #交叉验证
from sklearn.model_selection import GridSearchCV #搜索最优参数

def build_model(optimizer="adam",init="glorrot_uniform"):
    model = Sequential()
    model.add(Dense(12, kernel_initializer=init,input_dim=8, activation="relu"))
    model.add(Dense(8, kernel_initializer=init,activation="relu"))
    model.add(Dense(1, kernel_initializer=init,activation="sigmoid"))  # 输出
    # 1-10 5 ，6
    # 编译模型
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])
    return model


# 定于数据载入数据
filepath = r"C:\Users\Tsinghua-yincheng\Desktop\keras-Day3\pima-indians-diabetes.csv"
dataset = np.loadtxt(filepath, encoding="utf-8", delimiter=',')
X = dataset[:, 0:8]
y = dataset[:, 8]
# 设定随机数种子
np.random.seed(17)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)  # 随机打乱数据顺序


model=KerasClassifier(build_fn=build_model,verbose=0)


#构造调试参数，取得识别率最高-最优参数组合
param_grid={}
param_grid["optimizer"]=["adam","nadam","rmsprop"]
param_grid["epochs"]=[10,20,50,100,500]
param_grid["batch_size"]=[5,10,20]
param_grid["init"]=["glorot_uniform","uniform"]
grid=GridSearchCV(estimator=model,param_grid=param_grid)#搜索 最优
results=grid.fit(X,y) #训练

print("最优%f,适用参数%s"%(results.best_score_,results.best_params_))
means=results.cv_results_["mean_test_score"]
stds=results.cv_results_["std_test_score"]
params=results.cv_results_["params"]
for mean,std,param in zip(means,stds,params):
    print("%f,%f,%r"%(mean,std,param))





