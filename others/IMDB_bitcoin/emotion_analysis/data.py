# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import imdb
# (x_train,y_train),(x_validation,y_validation)=imdb.load_data()
# # print(x_train[:3])
# # print("---------------------")
# # print(y_train[:100])
#
# # print(x_train.shape)
# # print(y_train.shape)
# # print(len(x_validation))
#
# x = np.concatenate((x_train, x_validation), axis=0)
# y = np.concatenate((y_train, y_validation), axis=0)
# print(x_train[:1][0])
# print(len(x_train[:1][0]))
#
# print("_______________________________")
from keras.preprocessing import sequence
# MAX_WORDS = 500
# x_train = sequence.pad_sequences(x_train, maxlen=MAX_WORDS)
# x_validation = sequence.pad_sequences(x_validation, maxlen=MAX_WORDS)
#
# print(x_train[:3])
# print(x_train[:3].shape)
# print("x_train.shape=",x_train.shape)


# (x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=5000)
# # 合并数据集
# x = np.concatenate((x_train, x_validation), axis=0)
# y = np.concatenate((y_train, y_validation), axis=0)
# print(x.shape, y.shape)
# print(np.unique(y))
# # print(len(np.unique(np.stack(x))))#长度平均
# # 计算平均值与标准差
# result = [len(word) for word in x]
# print("mean  %f std %f" % (np.mean(result), np.std(result)))
#
# plt.subplot(121)
# plt.boxplot(result)
# plt.subplot(122)
# plt.hist(result)
# plt.show()

# list1 = [[1,5,1,9,5,3,7],
#          [5,9,7,1,2,3,6],
#          [6,5,7,1,2,8,4]]

list1 = [[1,5,1,9,5,3,7],
         [5,9,7,2,3],
         [6,5,7],
         [8,6,8,4,1,3]]

print(list1)

seq_list = sequence.pad_sequences(list1, maxlen=50)
print(seq_list)


# [[0 0 0 1 5 1 9 5 3 7]
#  [0 0 0 0 0 5 9 7 2 3]
#  [0 0 0 0 0 0 0 6 5 7]
#  [0 0 0 0 8 6 8 4 1 3]]
