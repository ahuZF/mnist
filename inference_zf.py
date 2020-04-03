import tensorflow as tf

def inference(input):
    with tf.variable_scope('conv1'):
        filter_weight=tf.get_variable('weights',[5,5,1,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases=tf.get_variable('biases',[32],initializer=tf.constant_initializer(0.1))

        conv=tf.nn.conv2d(input,filter_weight,[1,1,1,1],padding='SAME') #28*28*32
        conv1=tf.nn.relu(tf.nn.bias_add(conv,biases))

    with tf.name_scope('poo2'):
        pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID') #14*14*32

    with tf.variable_scope('conv3'):
        filter_weight=tf.get_variable('weights',[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases=tf.get_variable('biases',[64],initializer=tf.constant_initializer(0.1))

        conv=tf.nn.conv2d(pool1,filter_weight,[1,1,1,1],padding='SAME') #14*14*64
        conv2=tf.nn.relu(tf.nn.bias_add(conv,biases))

    with tf.name_scope('pool4'):
        pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID') #7*7*64

    with tf.variable_scope('fc1'):
        reshaped = tf.reshape(pool2, [100, 7*7*64])  ##[?,7,7,64], [3136,64].
        fcl_weights=tf.get_variable('weights',[7*7*64,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases=tf.get_variable('biases',[64],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fcl_weights)+biases)

    with tf.variable_scope('fc2'):
        fcl_weights = tf.get_variable('weights', [64, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [10], initializer=tf.constant_initializer(0.1))
        logit = tf.nn.relu(tf.matmul(fc1, fcl_weights)+ biases)
    return logit