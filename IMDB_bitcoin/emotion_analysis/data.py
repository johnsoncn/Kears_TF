

from keras.datasets import imdb
(x_train,y_train),(x_validation,y_validation)=imdb.load_data()
print(x_train)
print("---------------------")
print(y_train)

print(x_train.shape)
print(y_train.shape)