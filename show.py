import tensorflow as tf
import matplotlib.pyplot as plt

''' 读取MNIST数据方法一'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("G:\\ANACONDA\\envs\\Tensorflow\\MNIST_data",one_hot=True)

train_data = mnist.train.images   #所有训练数据
val_data = mnist.validation.images  #(5000,784)
test_data = mnist.test.images       #(10000,784)
plt.figure()
im = val_data[101].reshape(28,28)
plt.imshow(im, 'gray')
plt.pause(0.0000001)
