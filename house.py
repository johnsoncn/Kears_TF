from  sklearn import datasets
#导入数据
dataset=datasets.load_boston() #载入分类数据
X=dataset.data
y=dataset.target
print(X,y)


import numpy as np
from keras.models import Sequential  # 模型
from keras.layers import Dense  # 层
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split  # 数据切割
from keras.wrappers.scikit_learn import KerasRegressor #分类
from sklearn.model_selection import KFold #分类
from sklearn.model_selection import cross_val_score #交叉验证
from sklearn.model_selection import GridSearchCV #搜索最优参数
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed=7
np.random.seed(seed)

def build_model(units_list=[13] ,optimizer="adam",init="normal"):
    model = Sequential()
    unints=units_list[0]
    #外层输入层
    model.add(Dense(units=unints, kernel_initializer=init,input_dim=13, activation="relu"))
    for unints in units_list[1:]: #初始化中间层
        model.add(Dense(units=unints,activation="relu",kernel_initializer=init,))
    model.add(Dense(units=1,kernel_initializer=init)) #输出层

    # 1-10 5 ，6
    # 编译模型
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])
    return model



model=KerasRegressor(build_fn=build_model,epochs=10,batch_size=5,verbose=1)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(model,X,y,cv=kfold)
print("基本线 %f %f  MSE"%(results.mean(),results.std()))

#数据标准化
steps=[]
steps.append(("standardize",StandardScaler()))
steps.append(("mlp",model))
pipline=Pipeline(steps)

#构造调试参数，取得识别率最高-最优参数组合
param_grid={}
param_grid["units_list"]=[[20],[13,6]]
param_grid["optimizer"]=["adam"]
param_grid["epochs"]=[10,50]
param_grid["batch_size"]=[10]
param_grid["init"]=["glorot_uniform"]


#调试参数
scaler=StandardScaler()
scaler_x=scaler.fit_transform(X)
grid=GridSearchCV(estimator=model,param_grid=param_grid)#搜索 最优
results=grid.fit(X,y) #训练

print("最优%f,适用参数%s"%(results.best_score_,results.best_params_))
means=results.cv_results_["mean_test_score"]
stds=results.cv_results_["std_test_score"]
params=results.cv_results_["params"]
for mean,std,param in zip(means,stds,params):
    print("%f,%f,%r"%(mean,std,param))
