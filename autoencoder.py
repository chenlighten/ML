import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def glorot(fan_in, fan_out, constant  = 1):
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    low = -high
    return tf.random_uniform(shape = [fan_in, fan_out],
                              dtype = tf.float32, 
                              minval = low, maxval = high)

class AddictiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden1, n_hidden2, transfer_function = tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.weights = self._initialize_weights()
        
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden1 = tf.nn.relu((tf.add(tf.matmul(
                                    self.x + scale * tf.random_normal((self.n_input, )),
                                    self.weights['w1']), self.weights['b1'])))
        self.hidden2 = self.transfer(tf.add(tf.matmul(
                                    self.hidden1, self.weights['w2']), self.weights['b2']))
        self.reconstruction = tf.add(tf.matmul(self.hidden2, self.weights['w3']), self.weights['b3'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
                            self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        self.relative_cost = self.cost / tf.reduce_sum(self.x * self.x)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(glorot(self.n_input, self.n_hidden1))
        weights['b1'] = tf.Variable(tf.zeros([self.n_hidden1], dtype = tf.float32))
        weights['w2'] = tf.Variable(tf.zeros([self.n_hidden1, self.n_hidden2], dtype = tf.float32))
        weights['b2'] = tf.Variable(tf.zeros([self.n_hidden2], dtype = tf.float32))
        weights['w3'] = tf.Variable(tf.zeros([self.n_hidden2, self.n_input], dtype = tf.float32))
        weights['b3'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                   feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.relative_cost, feed_dict = {self.x: X, self.scale: self.training_scale})
    
    def transform(self, X):
        return self.sess.run(self.hidden1, feed_dict = {self.x: X, self.scale: self.training_scale})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden1: hidden})
    
    def reconstruction(self, X):
        return self.sess.run(reconstruction, feed_dict = {self.x: X, self.scale: self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBias(self):
        return self.sess.run(self.weights['b1'])

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(low = 0, high = len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

    N_SAMPLES = int(mnist.train.num_examples)
    TRAINING_EPOCHS = 20
    BATCH_SIZE = 256
    DISPLAY_DTEP = 1

    autoencoder = AddictiveGaussianNoiseAutoencoder(n_input = 784,
                                                    n_hidden1 = 200,
                                                    n_hidden2 = 400,
                                                    transfer_function = tf.nn.softplus,
                                                    optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3),
                                                    scale = 1e-2)
    
    writer = tf.summary.FileWriter("./logs/autoencoder/", autoencoder.sess.graph)
    for epoch in range(TRAINING_EPOCHS):
        avrg_cost = 0.0
        total_batch = int(N_SAMPLES / BATCH_SIZE)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, BATCH_SIZE)

            cost = autoencoder.partial_fit(batch_xs)
            avrg_cost += cost / N_SAMPLES * BATCH_SIZE
        
        if epoch % DISPLAY_DTEP == 0:
            print("Epoch: %04d" % (epoch + 1), "cost: %.9f" % avrg_cost)

    print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))