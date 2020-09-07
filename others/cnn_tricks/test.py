# -*- coding: utf-8 -*-
""" 
@Time    : 2020/8/25 10:38
@Author  : HCF
@FileName: test.py
@SoftWare: PyCharm
"""



from  sklearn import datasets
from keras.utils  import to_categorical
from keras.models import model_from_json
#导入数据
dataset=datasets.load_iris() #载入分类数据
X=dataset.data
y=dataset.target

print(X)
print(y)
print(X.shape)
print(y.shape)

#转化数据编码
Y_labels=to_categorical(y,num_classes=3)
print(Y_labels)