import tensorflow as tf
import numpy as np
import  inference_zf
from tensorflow.examples.tutorials.mnist import input_data

def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x-input')
    vs = mnist.validation.images[100:200]
    reshaped_vs = np.reshape(vs, (100, 28, 28, 1))
    validate_feed = {x: reshaped_vs}
    y=inference_zf.inference(x)
    correct_prediction=tf.argmax(y,1)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"G:\\PY\\Le_Net_zf\\save\\model.ckpt-30000")
        acc=sess.run(correct_prediction,feed_dict=validate_feed)
        print(acc)

def main(argv=None):
    mnist=input_data.read_data_sets("G:\\ANACONDA\\envs\\Tensorflow\\MNIST_data",one_hot=True)
    evaluate(mnist)

if __name__=='__main__':
    tf.app.run()