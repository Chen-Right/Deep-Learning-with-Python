# Note: 现实《Python深度学习》手写数字识别
#       模型训练、测试
# Date: 2021/3/5

import keras
from keras.datasets import mnist

# 导入数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 查看数据集信息
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
print(len(test_labels))
# 训练开始
from keras import models
from keras import layers
# 搭建神经网络层级
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
# 编译模型
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# 数据扁平化
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# 处理标签
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 拟合模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# 计算模型测试集误差
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


