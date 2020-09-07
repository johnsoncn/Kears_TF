
import numpy as np
from keras.models import Sequential,load_model  # 模型
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

#转化数据编码
Y_labels=to_categorical(y,num_classes=3)

print(X,y)
np.random.seed(17)

def build_model(optimeizer="adam",init="glorot_uniform"):
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation="relu",kernel_initializer=init))
    model.add(Dense(6, activation="relu",kernel_initializer=init))
    model.add(Dense(3, activation="softmax",kernel_initializer=init))  # 输出
    # 1-10 5 ，6
    model.compile(loss="categorical_crossentropy", optimizer=optimeizer, metrics=["accuracy"])
    return model

model=build_model()
model.fit(X,Y_labels,epochs=200,batch_size=5,verbose=1)#训练
scores=model.evaluate(X,Y_labels,verbose=0)#评分
print("%s %f" % (model.metrics_names[1],scores[1]*100))



# #保存模型为json文件
# model_json=model.to_json()
# with open("model.json","w") as file:
#     model_json=file.write(model_json)
#     print("model saved successfully")
#
# #保存模型权重
# model.save_weights("model_json.h5")


#加载模型
with open("model.json","r") as file:
    model_json=file.read()

new_model = model_from_json(model_json)#载入模型
new_model.load_weights("model_json.h5")

new_model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

#评估json加载模型
scores=new_model.evaluate(X,Y_labels,verbose=0)
print("new  %s %f" % (new_model.metrics_names[1],scores[1]*100))


