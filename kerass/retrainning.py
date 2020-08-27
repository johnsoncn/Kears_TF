

# 100条数据 模型X， 增量20条数据，反复训练，
#增量更新


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
#导入数据
dataset=datasets.load_iris() #载入分类数据
X=dataset.data
y=dataset.target
x_train,x_new,y_train,y_new=train_test_split(X,y,test_size=0.2,random_state=17)
#转化数据编码
Y_labels=to_categorical(y_train,num_classes=3)
print(X,y)
np.random.seed(17)

def build_model(optimeizer="adam",init="glorot_uniform"):
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation="relu",kernel_initializer=init))
    model.add(Dense(6, activation="relu",kernel_initializer=init))
    model.add(Dense(3, activation="softmax",kernel_initializer=init))  # 输出
    # 1-10 5 ，6
    # 编译模型
    model.compile(loss="categorical_crossentropy", optimizer=optimeizer, metrics=["accuracy"])
    return model

model=build_model()#建造模型
model.fit(x_train,Y_labels,epochs=200,batch_size=5,verbose=0)#训练

scores=model.evaluate(x_train,Y_labels,verbose=0)#评分

#保存模型为json文件
model_json=model.to_json()
with open("model.json","w") as file:
    model_json=file.write(model_json)#写入
#保存模型权重
model.save_weights("model_json.h5")

#加载模型
with open("model.json","r") as file:
    model_json=file.read()
new_model = model_from_json(model_json)#载入模型
new_model.load_weights("model_json.h5")

new_model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])


"""
新增数据，继续训练模型
"""
y_new_labels=to_categorical(y_new,num_classes=3)#增量
new_model.fit(x_new,y_new_labels,epochs=200,batch_size=5)

#评估json加载模型
scores1=new_model.evaluate(x_new,y_new_labels)
print("%s %f" % (model.metrics_names[1],scores[1]*100))
print("new  %s %f" % (new_model.metrics_names[1],scores1[1]*100))


