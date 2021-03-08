# Note: imdb数据集——训练正负面评论二分类模型
#       通过交叉验证集发现出现过拟合
# Date: 2021/3/5

from keras.datasets import imdb
# 导入数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# # 显示数据
# print(train_data[0])
# print(train_labels[0])
# print(max([max(sequence) for sequence in train_data]))

# # 解码为评论
# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decode_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
# print(decode_review)

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
# print(x_train[0])
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
"""4-留出交叉验证集数据"""
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
"""5-训练模型"""
history = model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=512,
          validation_data=(x_val,y_val))

# history_dict = history.history
# print(history_dict.keys())

"""绘制训练误差和验证误差曲线(学习曲线)"""
import matplotlib.pyplot as plt

history_dict = history.history
# 提取误差数据
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
# x轴
epochs = range(1, len(loss_values) + 1)
# 绘制误差曲线
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# 绘制准确性曲线
plt.clf()
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


