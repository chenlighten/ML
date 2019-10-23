from tensorflow.examples.tutorials.mnist import input_data
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

begin = time.clock()

n_hidden = 300
keep_prob = 0.75
learning_rate = 0.003
batch_size = 128
epochs = 5200

w1 = tf.Variable(tf.truncated_normal([784, n_hidden], dtype = tf.float32, stddev = 0.1))
b1 = tf.Variable(tf.zeros([n_hidden]))
w2 = tf.Variable(tf.zeros([n_hidden, 10]))
b2 = tf.Variable(tf.zeros(10))

keep = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 784])
hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(x, w1) + b1), keep_prob = keep)
y = tf.nn.softmax(tf.matmul(hidden, w2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_sum(-y_ * tf.log(y))
train_op = tf.train.AdamOptimizer().minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1)), tf.float32))

if __name__ == "__main__":
    sess = tf.Session()
    writer = tf.summary.FileWriter("./logs/efficientmnist/", sess.graph)
    sess.run(tf.global_variables_initializer())
    mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
    accs = []
    for epoch in range(epochs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict = {x: batch_x, y_: batch_y, keep: keep_prob})
        acc = sess.run(accuracy, feed_dict = {x: batch_x, y_: batch_y, keep: 1.0})
        accs.append(acc)
        print("epoch: %d"%epoch)

    acc = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep: 1.0})
    print(acc)
    print("time: %f"%(time.clock() - begin))
    plt.plot([i + 1 for i in range(epochs)], accs)
    plt.xlabel("Training Times")
    plt.ylabel("Accuracy")
    plt.show()
