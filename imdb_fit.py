# Note: 20迭代出现过拟合问题，通过学习曲线发现第四次最好
#       直接设置迭代次数为4，其他部分copy，不需要画曲线
#       同时不需要验证集了
# Data: 2021/3/5 19:24

from keras.datasets import imdb
# 导入数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

"""1-数据编码"""
import numpy as np
def vectorize_sequences (sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# # 显示编码数据
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

"""开始构建模型"""
from keras import models
from keras import layers
"""2-组建模型"""
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
"""3-编译模型"""
# # 方法1：作为字符串传入
# model.compile(optimizer='rmsporp',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# 方法2：传入函数对象，并且可以设置自定义优化器的参数
from keras import optimizers
from keras import losses
from keras import metrics
# 本人更倾向于第二种方法
model.compile(optimizer=optimizers.RMSprop(0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

"""4-训练模型"""
"""更改epochs=4"""
history = model.fit(x_train,
          y_train,
          epochs=4,
          batch_size=512)
results = model.evaluate(x_test, y_test)



