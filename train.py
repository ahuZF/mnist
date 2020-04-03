import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import inference_zf
training_steps = 30000
def train(mnist):
    x= tf.placeholder(tf.float32, [None, 28, 28, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    y=inference_zf.inference(x)
    global_step_train = tf.Variable(0, trainable=False)

    ##损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #学习率下降
    learning_rate = tf.train.exponential_decay(0.01, global_step_train, mnist.train.num_examples / 100, 0.99)
    #梯度下降
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean,global_step=global_step_train)
    ## 验证模型精度
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        vs=mnist.validation.images[:100]
        reshaped_vs = np.reshape(vs, (100, 28, 28, 1))
        validate_feed={x:reshaped_vs,y_:mnist.validation.labels[:100]}
        ts = mnist.test.images[:100]
        reshaped_ts = np.reshape(ts, (100, 28, 28, 1))
        test_feed = {x: reshaped_ts, y_: mnist.test.labels[:100]}

        for i in range(training_steps):
            xs, ys = mnist.train.next_batch(100)
            reshaped_xs = np.reshape(xs, (100, 28, 28, 1))
            _,step=sess.run([train_step,global_step_train], feed_dict={x: reshaped_xs, y_: ys})
            if i % 1000 == 0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s),validation accuracy on training batch is %g." % (step, validate_acc))
                print("当前学习率 %f,当前轮数 %d" %(learning_rate.eval(session=sess),step))
        saver.save(sess,os.path.join(r"G:\PY\Le_Net_zf\save","model.ckpt"),global_step=global_step_train)
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s),test accuracy on training batch is %g." % (training_steps, test_acc))

def main(argv=None):
    mnist=input_data.read_data_sets("G:\\ANACONDA\\envs\\Tensorflow\\MNIST_data",one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()