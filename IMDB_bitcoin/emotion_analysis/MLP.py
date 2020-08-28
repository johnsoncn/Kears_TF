from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.preprocessing import *
from keras.models import Sequential

# 构造MLP，多层感知机
SEED = 7  # 随机数种子
MAX_FEATURES = 5000
MAX_WORDS = 500
OUTPUT_DIM = 2  # 正面，负面
BATCH_SIZE = 128  # 批量处理
EPOCHS = 10  # 训练测试


def create_model():
    model = Sequential()
    # 包含层
    model.add(Embedding(MAX_FEATURES, OUTPUT_DIM, input_length=MAX_WORDS))
    model.add(Flatten())  # 数据展平
    model.add(Dense(250, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    model.summary()
    return model
    pass


if __name__ == "__main__":
    (x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=MAX_FEATURES)
    # 合并数据集
    x = np.concatenate((x_train, x_validation), axis=0)
    y = np.concatenate((y_train, y_validation), axis=0)
    print(x.shape, y.shape)
    print(np.unique(y))
    # print(len(np.unique(np.stack(x))))#长度平均
    # 计算平均值与标准差
    result = [len(word) for word in x]
    print("mean  %f std %f" % (np.mean(result), np.std(result)))

    # plt.subplot(121)
    # plt.boxplot(result)
    # plt.subplot(122)
    # plt.hist(result)
    # plt.show()

    # 数据填充
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_WORDS)
    x_validation = sequence.pad_sequences(x_validation, maxlen=MAX_WORDS)

    model = create_model()
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=BATCH_SIZE, epochs=EPOCHS,verbose=1)
    print("----------start evaluation-------------")
    scores = model.evaluate(x_validation, y_validation, verbose=1)
    print(scores)

    model.save("emotion_MLP.h5")
