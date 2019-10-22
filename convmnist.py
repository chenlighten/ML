import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 10000
BATCH_SIZE = 64
KEEP_PROB = 0.5
TEST_INTERVAL = 100
LEARNING_RATE = 1e-4

def createWeight(shape):
    return tf.Variable(tf.truncated_normal(shape, dtype = tf.float32, stddev = 0.1))

def createBias(shape):
    return tf.Variable(tf.constant(0.1, dtype = tf.float32, shape = shape))

def createConv(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')

def createPool(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

with tf.name_scope("Inputs"):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("First_conv"):
    w_conv1 = createWeight([5, 5, 1, 32])
    b_conv1 = createBias([32])
    h_conv1 = tf.nn.relu(createConv(x_image, w_conv1) + b_conv1)
    h_pool1 = createPool(h_conv1)

with tf.name_scope("Second_conv"):
    w_conv2 = createWeight([5, 5, 32, 64])
    b_conv2 = createBias([64])
    h_conv2 = tf.nn.relu(createConv(h_pool1, w_conv2) + b_conv2)
    h_pool2 = createPool(h_conv2)

with tf.name_scope("Dense"):
    w_fc1 = createWeight([7*7*64, 1024])
    b_fc1 = createBias([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

with tf.name_scope("Drop_out"):
    h_fc1_drop = tf.nn.dropout(h_fc1, rate = 1 - keep_prob)

with tf.name_scope("Output"):
    w_fc2 = createWeight([1024, 10])
    b_fc2 = createBias([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(-y_ * y, 1))

with tf.name_scope("Test"):
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

with tf.name_scope("Train"):
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

if __name__ == "__main__":
    sess = tf.Session()
    writer = tf.summary.FileWriter("./logs/convmnist/", sess.graph)
    sess.run(tf.global_variables_initializer())
    mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
    accuracies = []
    for epoch in range(EPOCHS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_op, feed_dict = {x: batch_x, y_: batch_y, keep_prob: KEEP_PROB})
        
        if epoch % TEST_INTERVAL == 0:
            acc = sess.run(accuracy, feed_dict = 
                {x: batch_x, y_: batch_y, keep_prob: 1})
            print("step %d, accuracy %f"%(epoch, acc))
            accuracies.append(acc)

    acc = sess.run(accuracy, feed_dict = 
        {x: mnist.test.images[0:5000], y_: mnist.test.labels[0:5000], keep_prob: 1.0})
    plt.plot(accuracies)
    plt.show()
    print("Total accuracy %f" % acc)
    print("\n\n\n")