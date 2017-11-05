import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets import mnist as input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
images, labels = mnist.train.next_batch(10)
print(images[0])
print(labels[0])

fig = plt.figure(figsize=(8,4))
for c, (image, label) in enumerate(zip(images, labels)):
    subplot = fig.add_subplot(2,5,c+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(label))
    subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1,cmap=plt.cm.gray_r, interpolation="nearest")