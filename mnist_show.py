# Note: mnist数据集数据显示
# Date: 2021/3/5

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


