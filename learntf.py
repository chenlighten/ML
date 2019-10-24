import tensorflow as tf
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

LEATNING_RATE = 0.01
TRAINING_TIME = 30000

def train_mnist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_op = tf.train.GradientDescentOptimizer(LEATNING_RATE).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    merged_summary_op = tf.summary.merge_all()
    
    sess = tf.Session()
    sess.run(init)
    for _ in range(TRAINING_TIME):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
        # if _ % 100 == 0:
        #     summary_str = sess.run(merged_summary_op)
        #     summary_writer.add_summary(summary_str, _)
    summary_writer = tf.summary.FileWriter('/tmp/mnist_logs', sess.graph)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                            strides=[1, 2, 2, 1], padding='SAME')

def train_mnist_effecient():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_mean(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correst_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correst_prediction, tf.float32))
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = sess.run(accuracy, 
                feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%(sess.run(accuracy,
        feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))


def choose_device():
    config = tf.ConfigProto(log_device_placement=True)
    with tf.device("/cpu:0"):
        rand_unif = tf.random_uniform([300, 3000], 0, 10)
        rand_norm = tf.random_normal([3000, 300])
        a = tf.Variable(rand_unif)
        b = tf.Variable(rand_norm)
    with tf.device("/cpu:0"):
        c = tf.matmul(a, b)
        init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(c))
    sess.close()

def test_device():
    # 新建一个 graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # 新建session with log_device_placement并设置为True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # 运行这个 op.
    print(sess.run(c))

def linear_regression():
    M = 20
    N = 15
    S = 100
    W_0 = np.random.random((M, N))
    def f(x):
        return np.dot(W_0, x.reshape((N, 1)))
    x = tf.placeholder('float', (N, 1))
    y_ = tf.placeholder('float', (M, 1))
    W = tf.Variable(initial_value=tf.constant(0.01, shape=(M, N)),  dtype='float')
    y = tf.matmul(W, tf.reshape(x, (N, 1)))
    loss = tf.reduce_mean(tf.square(y - y_))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    init_op = tf.initialize_all_variables()

    err = 1 - tf.divide(y, y_)
    max_err = tf.reduce_max(err)

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(1000000):
            x_data = np.random.random((N, 1))
            y_data = f(x_data)
            sess.run(train_op, feed_dict={x: x_data, y_: y_data})
        errs = []
        for i in range(10):
            x_data = np.random.random((N, 1))
            y_data = f(x_data)
            errs.append(np.max(sess.run(max_err, feed_dict={x: x_data, y_: y_data})))
        print(errs)

    X = np.random.random((N, S))
    Y = np.zeros((M, 0))
    for i in range(S):
        Y = np.concatenate((Y, f(X[:, i])), axis=1)
    # 想办法找个公式直接求出回归方程
    W_theo = np.matmul(np.matmul(Y, X.T), np.linalg.inv(np.matmul(X, X.T)))
    errs = []
    for i in range(10):
        x_data = np.random.random((N, 1))
        y_data_real = f(x_data)
        y_data_pred = np.dot(W_theo, x_data)
        errs.append(np.max(1 - y_data_pred/y_data_real))
    print(errs)


    
    
    
        
    

if __name__ == "__main__":
    # train_mnist()
    # test_tensorboard()
    train_mnist_effecient()
    # choose_device()
    # test_device()
    # linear_regression()