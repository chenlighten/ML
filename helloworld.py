import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01
LOOPS = 1000
BATCH_SIZE = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

x = tf.placeholder(tf.float32, [None, 784], "inputs")
W = tf.Variable(tf.zeros([784, 10]), name = "weights")
b = tf.Variable(tf.zeros([10]), name = "bias")
y = tf.nn.softmax(tf.matmul(x, W) + b, name = "output")

y_ = tf.placeholder(tf.float32, [None, 10], name = "real")
cross_entropy_tensor = -y_*tf.log(y)
cross_entropy = tf.reduce_sum(cross_entropy_tensor, name = "loss")
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy, name = "train")

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32), name = "accuracy")

if __name__ == '__main__':
    accs = []
    sess = tf.Session()
    writer = tf.summary.FileWriter("./logs/helloworld/", sess.graph)
    sess.run(tf.global_variables_initializer())
    for _ in range(LOOPS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_op, feed_dict = {x: batch_x, y_: batch_y})
        acc = sess.run(accuracy, feed_dict = {x: batch_x, y_: batch_y})
        accs.append(acc)
    acc = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
    print(acc)
    plt.plot([i + 1 for i in range(LOOPS)], accs)
    plt.xlabel("Training Times")
    plt.ylabel("Accuracy")
    plt.show()

    
    