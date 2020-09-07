

from  keras.preprocessing.image import ImageDataGenerator #处理图像数据
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt

cats=r"train/cat"
dogs=r"train/dog"
# print(len(os.listdir(cat_path)))
# print(os.listdir(dog_path))
# print(os.listdir(cat_path)[3])

cat_0 = image.load_img(cats+'/cat.1.jpg')
plt.imshow(cat_0)
plt.show()

print("format={}".format(cat_0.format))  # JPEG
print("mode={}".format(cat_0.mode))   #RGB
print("size={}".format(cat_0.size))   #(500,537)
print("info={}".format(cat_0.info))  #以字典形式返回图片信息

cats_array = image.img_to_array(cat_0)
print("shape = {}".format(cats_array.shape))