from  keras.preprocessing.image import ImageDataGenerator #处理图像数据
import os
from keras.preprocessing import  image
import matplotlib.pyplot as plt

cat_path=r"train/cat"
dog_path=r"train/dog"
print(len(os.listdir(cat_path)))
print(os.listdir(dog_path))



train_datagen=ImageDataGenerator(rescale=1./255) #图像实例化
test_datagen=ImageDataGenerator(rescale=1./255)


train_dir=r"train"
test_dir=r"test"

# flow_from_directory获取图片
train_Gen=train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),# 设置图片大小（文件夹原图大小可能不一样，先统一图片大小，那么网络就需要输入150，150，3）
    batch_size=20, # 批次大小，每次就会从路径下取出20张，并统一为150*150大小
    class_mode="binary")
vali_Gen=train_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode="binary")

imgpath=train_dir+"\\"+os.listdir(cat_path)[3]#//取得第三章图片
print(imgpath)
img=image.load_img(imgpath,target_size=(150,150))
x=image.img_to_array(img)
x=x.reshape((1,)+x.shape) #//[1,150,150,3]
plt.figure(figsize=[12,12])
i=0

datagen=ImageDataGenerator(
    rotation_range=40,#图片旋转处理
    width_shift_range=0.2,
    height_shift_range=0.2,#水平方向垂直方向的随机为转移
    shear_range=0.2, #错切变换
    zoom_range=0.2, #缩放
    horizontal_flip=True, #水平旋转
    fill_mode="nearest"#填充创建像素的方法
)
for batch in datagen.flow(x,batch_size=1):
    plt.subplot(2,2,i+1)
    imglot=plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i%4==0:
        break
plt.show()



for data_batch,label_batch in train_Gen:
    print(data_batch.shape)
    print(label_batch)
    #print(data_batch.shape)
    #print(label_batch.shape)