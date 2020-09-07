
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
import matplotlib.pyplot as plt

#导入数据
dataset=datasets.load_iris() #载入分类数据
X=dataset.data
y=dataset.target

#转化数据编码
Y_labels=to_categorical(y,num_classes=3)

print(X.shape,y.shape)
np.random.seed(17)

def build_model(optimeizer="adam",init="glorot_uniform"):
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation="relu",kernel_initializer=init))
    model.add(Dense(6, activation="relu",kernel_initializer=init))
    model.add(Dense(3, activation="softmax",kernel_initializer=init))  # 输出

    model.compile(loss="categorical_crossentropy", optimizer=optimeizer, metrics=["accuracy"])
    model.summary()
    return model

model=build_model()

trained_model = model.fit(X,Y_labels,validation_split=0.2,epochs=200,batch_size=5,verbose=1)
# scores=model.evaluate(X,Y_labels,verbose=1)#评分
# print("%s %f" % (model.metrics_names[1],scores[1]*100))

print(trained_model.history.keys())
# #accuracy
plt.plot(trained_model.history["accuracy"])
plt.plot(trained_model.history["val_accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train","validation"],loc="upper left")
plt.show()


#loss
plt.plot(trained_model.history["loss"])
plt.plot(trained_model.history["val_loss"])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","validation"],loc="upper left")
plt.show()
